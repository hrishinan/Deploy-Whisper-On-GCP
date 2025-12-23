import os
import time
import json
import tempfile
import html
import urllib.parse
from datetime import datetime

import whisper
from fastapi import FastAPI, Request
from google.cloud import storage
from google.api_core.exceptions import NotFound

# =====================================================
# CONFIG
# =====================================================
MODEL_VERSION = "large-v3-turbo"
NUM_MELS = 128

GCS_INPUT_BUCKET = os.environ["GCS_INPUT_BUCKET"]
GCS_OUTPUT_BUCKET = os.environ["GCS_OUTPUT_BUCKET"]
GCS_OUTPUT_PREFIX = os.environ.get("GCS_OUTPUT_PREFIX", "whisper-results")

# =====================================================
# APP + MODEL
# =====================================================
app = FastAPI()

MODEL_PATH = f"/app/models/{MODEL_VERSION}.pt"
MODEL = whisper.load_model(MODEL_PATH)

gcs_client = storage.Client()

# =====================================================
# LOGGING
# =====================================================
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

# =====================================================
# CORE WHISPER PIPELINE (ORIGINAL BEHAVIOR)
# =====================================================
def run_whisper_pipeline(audio_path: str, source_name: str, file_size_mb: float):
    pipeline_start = time.time()

    log(f"[PIPELINE START]")
    log(f"Input Audio Sample : {source_name}")
    log(f"File Size (MB)     : {file_size_mb:.2f}")

    # -------------------------------------------------
    # Decode audio (ORIGINAL PATH – FFmpeg via Whisper)
    # -------------------------------------------------
    decode_start = time.time()
    audio = whisper.load_audio(audio_path)
    sr = whisper.audio.SAMPLE_RATE
    decode_time = time.time() - decode_start

    duration_sec = len(audio) / sr

    log(f"Audio Duration (s) : {duration_sec:.2f}")
    log(f"Decode Time (s)    : {decode_time:.2f}")

    # -------------------------------------------------
    # ORIGINAL SINGLE-PASS WHISPER INFERENCE
    # -------------------------------------------------
    infer_start = time.time()
    mel = whisper.log_mel_spectrogram(audio, n_mels=NUM_MELS).to(MODEL.device)
    result = whisper.decode(MODEL, mel)
    inference_time = time.time() - infer_start

    log(f"Inference Time (s) : {inference_time:.2f}")

    total_time = time.time() - pipeline_start

    # -------------------------------------------------
    # Build result JSON (unchanged semantics)
    # -------------------------------------------------
    output = {
        "input_audio": source_name,
        "file_size_mb": round(file_size_mb, 2),
        "duration_sec": round(duration_sec, 2),
        "inference_time_sec": round(inference_time, 2),
        "total_completion_time_sec": round(total_time, 2),
        "text": result.text,
        "language": result.language,
    }

    # -------------------------------------------------
    # Upload output to GCS
    # -------------------------------------------------
    upload_start = time.time()

    out_name = (
        f"{GCS_OUTPUT_PREFIX}/"
        f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
        f"{os.path.basename(source_name)}.json"
    )

    blob = gcs_client.bucket(GCS_OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(
        json.dumps(output, indent=2),
        content_type="application/json"
    )

    upload_time = time.time() - upload_start

    log(f"Output Uploaded    : gs://{GCS_OUTPUT_BUCKET}/{out_name}")
    log(f"Upload Time (s)    : {upload_time:.2f}")
    log(f"Total Completion  : {total_time:.2f}s")
    log(f"[PIPELINE COMPLETE]")

    return output

# =====================================================
# EVENTARC GCS TRIGGER
# =====================================================
@app.post("/gcs-trigger")
async def gcs_trigger(request: Request):
    event_start = time.time()

    event = await request.json()
    payload = event.get("data", event)

    raw_name = payload.get("name")
    bucket = payload.get("bucket")
    generation = payload.get("generation")
    size_bytes = int(payload.get("size", 0))

    # Normalize object name (NO behavior change, correctness only)
    name = urllib.parse.unquote(html.unescape(raw_name)) if raw_name else None
    file_size_mb = size_bytes / (1024 * 1024)

    log(f"[EVENT RECEIVED]")
    log(f"Bucket             : {bucket}")
    log(f"Raw Object Name    : {raw_name}")
    log(f"Decoded Name       : {name}")
    log(f"Generation         : {generation}")
    log(f"Reported Size (MB) : {file_size_mb:.2f}")

    if not bucket or not name or not generation:
        log("[EVENT IGNORED] Missing required fields")
        return {"status": "ignored"}

    if bucket != GCS_INPUT_BUCKET:
        log("[EVENT IGNORED] Unexpected bucket")
        return {"status": "ignored"}

    if not name.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        log("[EVENT IGNORED] Not an audio file")
        return {"status": "ignored"}

    # -------------------------------------------------
    # Download exact object generation
    # -------------------------------------------------
    download_start = time.time()

    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_audio = f.name

    blob = gcs_client.bucket(bucket).blob(name, generation=generation)

    try:
        blob.download_to_filename(temp_audio)
    except NotFound:
        log(
            f"[SKIPPED] Object no longer exists "
            f"(bucket={bucket}, name={name}, generation={generation})"
        )
        return {"status": "skipped", "reason": "object_not_found"}

    download_time = time.time() - download_start

    log(f"Download Time (s)  : {download_time:.2f}")
    log(f"Temp File Path    : {temp_audio}")

    try:
        run_whisper_pipeline(temp_audio, name, file_size_mb)
    finally:
        os.remove(temp_audio)
        log("Temp File Cleaned")

    log(f"[EVENT COMPLETE] Total Event Time: {time.time() - event_start:.2f}s")

    return {"status": "processed", "file": name}
