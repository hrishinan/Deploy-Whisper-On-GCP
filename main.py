import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
import subprocess
import tempfile
import shutil
import os
import torch
import time
import json
from datetime import datetime

from google.cloud import storage

# =====================================================
# CONFIG
# =====================================================
MODEL_VERSION = "large-v3-turbo"
NUM_MELS = 128

GCS_OUTPUT_BUCKET = os.environ.get("GCS_OUTPUT_BUCKET")
GCS_OUTPUT_PREFIX = os.environ.get("GCS_OUTPUT_PREFIX", "whisper-results")

if not GCS_OUTPUT_BUCKET:
    raise RuntimeError("GCS_OUTPUT_BUCKET env var is required")

# =====================================================
# APP + MODEL
# =====================================================
app = FastAPI()

MODEL_PATH = f"/app/models/{MODEL_VERSION}.pt"
MODEL = whisper.load_model(MODEL_PATH)

gcs_client = storage.Client()

# =====================================================
# HELPERS
# =====================================================
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)


def save_upload_file_to_temp(upload_file: UploadFile) -> str:
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        upload_file.file.seek(0)
        shutil.copyfileobj(upload_file.file, temp_file)
        return temp_file.name


def upload_json_to_gcs(data: dict, object_name: str):
    bucket = gcs_client.bucket(GCS_OUTPUT_BUCKET)
    blob = bucket.blob(object_name)
    blob.upload_from_string(
        json.dumps(data, indent=2),
        content_type="application/json",
    )


# =====================================================
# CORE WHISPER PIPELINE (REUSED)
# =====================================================
def run_whisper_pipeline(audio_path: str, source_name: str):
    response = {}
    start_wall = time.time()

    # Load audio
    audio = whisper.load_audio(audio_path)
    audio_duration = len(audio) / whisper.audio.SAMPLE_RATE
    response["audio_duration_sec"] = round(audio_duration, 2)

    # Pad / trim
    audio = whisper.pad_or_trim(audio)

    # Mel features
    mel = whisper.log_mel_spectrogram(
        audio, n_mels=NUM_MELS
    ).to(MODEL.device)

    # Inference
    s = time.time()
    result = whisper.decode(MODEL, mel)
    inference_time = time.time() - s

    throughput = audio_duration / inference_time if inference_time > 0 else 0

    response.update({
        "source": source_name,
        "text": result.text,
        "language": result.language,
        "model": MODEL_VERSION,
        "device": str(MODEL.device),
        "inference_time_sec": round(inference_time, 3),
        "throughput_audio_sec_per_sec": round(throughput, 2),
        "wall_time_total_sec": round(time.time() - start_wall, 2),
    })

    # Upload output
    object_name = (
        f"{GCS_OUTPUT_PREFIX}/"
        f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
        f"{os.path.basename(source_name)}_{int(start_wall)}.json"
    )

    upload_json_to_gcs(response, object_name)

    log(
        f"Processed {source_name} | "
        f"Inference {inference_time:.2f}s | "
        f"Throughput {throughput:.2f}x | "
        f"Output gs://{GCS_OUTPUT_BUCKET}/{object_name}"
    )

    return response


# =====================================================
# HEALTH CHECKS
# =====================================================
@app.post("/check-gpu/")
async def check_gpu():
    if not torch.cuda.is_available():
        raise HTTPException(status_code=400, detail="CUDA is not available")
    return {"cuda": True}


@app.post("/check-ffmpeg/")
async def check_ffmpeg():
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            check=True
        )
    except Exception:
        raise HTTPException(status_code=400, detail="FFMPEG is not available")
    return {"ffmpeg": True}


@app.post("/check-model-in-memory/")
async def check_model_in_memory():
    return {"contents": os.listdir("/app/models/")}


# =====================================================
# HTTP API (MANUAL UPLOAD)
# =====================================================
@app.post("/translate/")
async def translate(file: UploadFile = File(...)):
    temp_path = save_upload_file_to_temp(file)
    try:
        return run_whisper_pipeline(temp_path, file.filename)
    finally:
        os.remove(temp_path)


# =====================================================
# GCS EVENTARC TRIGGER
# =====================================================
@app.post("/gcs-trigger")
async def gcs_trigger(request: Request):
    event = await request.json()
    data = event.get("data", {})

    bucket_name = data.get("bucket")
    object_name = data.get("name")
    size = data.get("size")

    log(f"GCS event: gs://{bucket_name}/{object_name} ({size} bytes)")

    # Skip non-audio files
    if not object_name.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        log("Ignored non-audio file")
        return {"status": "ignored"}

    # Download file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_audio = f.name

    gcs_client.bucket(bucket_name).blob(object_name).download_to_filename(temp_audio)

    try:
        run_whisper_pipeline(temp_audio, object_name)
    finally:
        os.remove(temp_audio)

    return {"status": "processed", "file": object_name}
