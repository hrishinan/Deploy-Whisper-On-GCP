import os
import time
import json
import tempfile
from datetime import datetime

import torch
import whisper
import numpy as np

from fastapi import FastAPI, Request
from google.cloud import storage
from silero_vad import load_silero_vad, get_speech_timestamps

# =====================================================
# CONFIG
# =====================================================
MODEL_VERSION = "large-v3-turbo"
NUM_MELS = 128

GCS_INPUT_BUCKET = os.environ["GCS_INPUT_BUCKET"]
GCS_OUTPUT_BUCKET = os.environ["GCS_OUTPUT_BUCKET"]
GCS_OUTPUT_PREFIX = os.environ.get("GCS_OUTPUT_PREFIX", "whisper-results")

USE_VAD = os.getenv("USE_VAD", "false").lower() == "true"

# =====================================================
# APP + MODEL
# =====================================================
app = FastAPI()

MODEL_PATH = f"/app/models/{MODEL_VERSION}.pt"
MODEL = whisper.load_model(MODEL_PATH)

gcs_client = storage.Client()
VAD_MODEL = load_silero_vad()

# =====================================================
# LOGGING
# =====================================================
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)

# =====================================================
# OPTIONAL HELPERS (NO BEHAVIOUR CHANGE)
# =====================================================
def apply_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    speech_ts = get_speech_timestamps(audio, VAD_MODEL, sampling_rate=sr)
    if not speech_ts:
        return audio
    return np.concatenate([audio[t["start"]:t["end"]] for t in speech_ts])

def chunk_audio(audio: np.ndarray, sr: int, chunk_sec: int = 60):
    samples = chunk_sec * sr
    return [audio[i:i + samples] for i in range(0, len(audio), samples)]

# =====================================================
# CORE WHISPER PIPELINE (BEHAVIOUR SAFE)
# =====================================================
def run_whisper_pipeline(audio_path: str, source_name: str, file_size_mb: float):
    pipeline_start = time.time()

    log(f"[PIPELINE START] file={source_name} size={file_size_mb:.2f}MB")

    # -----------------------------
    # Decode audio (FFmpeg)
    # -----------------------------
    decode_start = time.time()
    audio = whisper.load_audio(audio_path)
    sr = whisper.audio.SAMPLE_RATE
    decode_time = time.time() - decode_start

    duration_sec = len(audio) / sr
    log(f"[AUDIO DECODED] duration={duration_sec:.2f}s decode_time={decode_time:.2f}s")

    # -----------------------------
    # Optional VAD (off by default)
    # -----------------------------
    if USE_VAD:
        vad_start = time.time()
        audio = apply_vad(audio, sr)
        log(f"[VAD DONE] time={time.time() - vad_start:.2f}s")

        chunks = chunk_audio(audio, sr)
    else:
        chunks = [audio]

    log(f"[CHUNKING] chunks={len(chunks)}")

    # -----------------------------
    # Whisper inference
    # -----------------------------
    infer_start = time.time()
    texts = []

    for idx, chunk in enumerate(chunks, 1):
        log(f"[INFER] chunk={idx}/{len(chunks)} start")
        mel = whisper.log_mel_spectrogram(chunk, n_mels=NUM_MELS).to(MODEL.device)
        out = whisper.decode(MODEL, mel)
        texts.append(out.text)

    inference_time = time.time() - infer_start
    log(f"[INFERENCE DONE] time={inference_time:.2f}s")

    # -----------------------------
    # Final result
    # -----------------------------
    total_time = time.time() - pipeline_start

    result = {
        "input_audio": source_name,
        "file_size_mb": round(file_size_mb, 2),
        "duration_sec": round(duration_sec, 2),
        "chunks": len(chunks),
        "inference_time_sec": round(inference_time, 2),
        "total_completion_time_sec": round(total_time, 2),
        "text": " ".join(texts),
    }

    # -----------------------------
    # Upload result
    # -----------------------------
    upload_start = time.time()

    out_name = (
        f"{GCS_OUTPUT_PREFIX}/"
        f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
        f"{os.path.basename(source_name)}.json"
    )

    blob = gcs_client.bucket(GCS_OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(json.dumps(result, indent=2), content_type="application/json")

    upload_time = time.time() - upload_start
    log(f"[UPLOAD DONE] gs://{GCS_OUTPUT_BUCKET}/{out_name} upload_time={upload_time:.2f}s")

    log(
        f"[PIPELINE COMPLETE] "
        f"file={source_name} "
        f"size={file_size_mb:.2f}MB "
        f"duration={duration_sec:.2f}s "
        f"total_time={total_time:.2f}s"
    )

    return result

# =====================================================
# EVENTARC GCS TRIGGER
# =====================================================
@app.post("/gcs-trigger")
async def gcs_trigger(request: Request):
    event = await request.json()
    payload = event.get("data", event)

    bucket = payload.get("bucket")
    name = payload.get("name")
    generation = payload.get("generation")
    size_bytes = int(payload.get("size", 0))
    file_size_mb = size_bytes / (1024 * 1024)

    log(f"[EVENT RECEIVED] bucket={bucket} object={name} generation={generation}")

    if not bucket or not name or not generation:
        log("[EVENT IGNORED] invalid payload")
        return {"status": "ignored"}

    if bucket != GCS_INPUT_BUCKET:
        log(f"[EVENT IGNORED] unexpected bucket {bucket}")
        return {"status": "ignored"}

    if not name.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        log(f"[EVENT IGNORED] non-audio file {name}")
        return {"status": "ignored"}

    # -----------------------------
    # Download exact generation
    # -----------------------------
    download_start = time.time()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_audio = f.name

    blob = gcs_client.bucket(bucket).blob(name, generation=generation)
    blob.download_to_filename(temp_audio)

    log(f"[DOWNLOAD DONE] time={time.time() - download_start:.2f}s")

    try:
        run_whisper_pipeline(temp_audio, name, file_size_mb)
    finally:
        os.remove(temp_audio)
        log("[TEMP FILE CLEANED]")

    return {"status": "processed", "file": name}
