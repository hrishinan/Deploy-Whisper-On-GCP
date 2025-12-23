import os
import time
import json
import tempfile
from datetime import datetime

import numpy as np
import soundfile as sf
import torch
import whisper

from fastapi import FastAPI, Request
from google.cloud import storage

# ================================
# NEW: VAD imports (RunPod parity)
# ================================
from silero_vad import load_silero_vad, get_speech_timestamps


# =====================================================
# CONFIG
# =====================================================
MODEL_VERSION = "large-v3-turbo"
NUM_MELS = 128
CHUNK_SEC = 60  # RunPod uses ~60s chunks

GCS_INPUT_BUCKET = os.environ["GCS_INPUT_BUCKET"]
GCS_OUTPUT_BUCKET = os.environ["GCS_OUTPUT_BUCKET"]
GCS_OUTPUT_PREFIX = os.environ.get("GCS_OUTPUT_PREFIX", "whisper-results")

# =====================================================
# APP + MODELS
# =====================================================
app = FastAPI()

MODEL_PATH = f"/app/models/{MODEL_VERSION}.pt"
MODEL = whisper.load_model(MODEL_PATH)

# ================================
# NEW: Load VAD once (global)
# ================================
VAD_MODEL = load_silero_vad()

gcs_client = storage.Client()


# =====================================================
# LOGGING
# =====================================================
def log(msg: str):
    print(f"[{datetime.utcnow().isoformat()}] {msg}", flush=True)


# =====================================================
# NEW: VAD + CHUNKING HELPERS
# =====================================================
def apply_vad(audio: np.ndarray, sr: int) -> np.ndarray:
    """
    Removes silence using Silero VAD.
    Matches RunPod: 'VAD filter removed XX sec'
    """
    speech_ts = get_speech_timestamps(audio, VAD_MODEL, sampling_rate=sr)

    if not speech_ts:
        return audio

    voiced = []
    for ts in speech_ts:
        voiced.append(audio[ts["start"]:ts["end"]])

    return np.concatenate(voiced)


def chunk_audio(audio: np.ndarray, sr: int, chunk_sec: int = CHUNK_SEC):
    """
    Splits audio into <=60s chunks.
    Matches RunPod chunking behavior.
    """
    samples = chunk_sec * sr
    return [
        audio[i:i + samples]
        for i in range(0, len(audio), samples)
    ]


# =====================================================
# NEW: RUNPOD-STYLE WHISPER PIPELINE
# =====================================================
def run_whisper_pipeline(audio_path: str, source_name: str):
    start = time.time()

    # -----------------------------
    # FIX: Use FFmpeg-based decoder
    # -----------------------------
    audio = whisper.load_audio(audio_path)
    sr = whisper.audio.SAMPLE_RATE

    original_duration = len(audio) / sr

    # -----------------------------
    # VAD
    # -----------------------------
    vad_start = time.time()
    audio = apply_vad(audio, sr)
    vad_time = time.time() - vad_start

    # -----------------------------
    # Chunking
    # -----------------------------
    chunks = chunk_audio(audio, sr)

    log(f"Audio {original_duration:.2f}s → {len(chunks)} chunks after VAD")

    texts = []
    infer_time = 0.0

    for i, chunk in enumerate(chunks, 1):
        log(f"Processing chunk {i}/{len(chunks)}")

        mel = whisper.log_mel_spectrogram(chunk, n_mels=NUM_MELS).to(MODEL.device)

        s = time.time()
        out = whisper.decode(MODEL, mel)
        infer_time += time.time() - s

        texts.append(out.text)

    total_time = time.time() - start
    throughput = original_duration / infer_time if infer_time > 0 else 0

    result = {
        "source": source_name,
        "audio_duration_sec": round(original_duration, 2),
        "chunks": len(chunks),
        "vad_time_sec": round(vad_time, 2),
        "inference_time_sec": round(infer_time, 2),
        "throughput_x_real_time": round(throughput, 2),
        "total_wall_time_sec": round(total_time, 2),
        "text": " ".join(texts),
    }

    # -----------------------------
    # Upload result to GCS
    # -----------------------------
    out_name = (
        f"{GCS_OUTPUT_PREFIX}/"
        f"{datetime.utcnow().strftime('%Y/%m/%d')}/"
        f"{os.path.basename(source_name)}.json"
    )

    blob = gcs_client.bucket(GCS_OUTPUT_BUCKET).blob(out_name)
    blob.upload_from_string(json.dumps(result, indent=2), content_type="application/json")

    log(f"Uploaded result to gs://{GCS_OUTPUT_BUCKET}/{out_name}")

    return result


# =====================================================
# GCS EVENTARC HANDLER
# =====================================================
@app.post("/gcs-trigger")
async def gcs_trigger(request: Request):
    event = await request.json()

    # Support both CloudEvents + legacy GCS payloads
    payload = event.get("data", event)

    bucket = payload.get("bucket")
    name = payload.get("name")

    if not bucket or not name:
        log(f"Ignoring invalid event: {event}")
        return {"status": "ignored"}

    if bucket != GCS_INPUT_BUCKET:
        log(f"Ignoring event from bucket {bucket}")
        return {"status": "ignored"}

    if not name.lower().endswith((".mp3", ".wav", ".m4a", ".flac")):
        log(f"Ignoring non-audio file {name}")
        return {"status": "ignored"}

    # Download file
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_audio = f.name

    gcs_client.bucket(bucket).blob(name).download_to_filename(temp_audio)

    try:
        run_whisper_pipeline(temp_audio, name)
    finally:
        os.remove(temp_audio)

    return {"status": "processed", "file": name}
