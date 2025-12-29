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
# CORE WHISPER PIPELINE
# (MATCHES ORIGINAL GITHUB BEHAVIOR)
# =====================================================
def run_whisper_pipeline(audio_path: str, source_name: str, file_size_mb: float):
    pipeline_start = time.time()
    timings = {}

    log("========== PIPELINE START ==========")
    log(f"Input Audio            : {source_name}")
    log(f"File Size (MB)         : {file_size_mb:.2f}")

    # -------------------------------------------------
    # Load audio
    # -------------------------------------------------
    t0 = time.time()
    audio = whisper.load_audio(audio_path)
    timings["load_audio_sec"] = time.time() - t0

    duration_sec = len(audio) / whisper.audio.SAMPLE_RATE
    log(f"Original Duration (s)  : {duration_sec:.2f}")
    log(f"Load Audio Time (s)    : {timings['load_audio_sec']:.2f}")

    # -------------------------------------------------
    # Pad / Trim (REQUIRED BY ORIGINAL CODE)
    # -------------------------------------------------
    t0 = time.time()
    audio = whisper.pad_or_trim(audio)
    timings["pad_trim_sec"] = time.time() - t0
    log(f"Pad/Trim Time (s)      : {timings['pad_trim_sec']:.2f}")

    # -------------------------------------------------
    # Compute MEL features (EXPLICIT)
    # -------------------------------------------------
    t0 = time.time()
    mel = whisper.log_mel_spectrogram(
        audio, n_mels=NUM_MELS
    ).to(MODEL.device)
    timings["mel_compute_sec"] = time.time() - t0
    log(f"Mel Compute Time (s)   : {timings['mel_compute_sec']:.2f}")

    # -------------------------------------------------
    # Decode (FORCED TRANSLATION — ORIGINAL BEHAVIOR)
    # -------------------------------------------------
    log("Starting Whisper decode (translate → English)")

    decode_options = whisper.DecodingOptions(
        task="translate",
        language="en",
        fp16=True
    )

    t0 = time.time()
    result = whisper.decode(MODEL, mel, decode_options)
    timings["decode_sec"] = time.time() - t0

    log(f"Decode Time (s)        : {timings['decode_sec']:.2f}")
    log(f"Detected Language      : {result.language}")

    total_time = time.time() - pipeline_start

    # -------------------------------------------------
    # Build output JSON (summary-friendly)
    # -------------------------------------------------
    output = {
        "input_audio": source_name,
        "file_size_mb": round(file_size_mb, 2),
        "duration_sec": round(duration_sec, 2),
        "language_detected": result.language,
        "text": result.text.strip(),
        "timings": {
            "load_audio_sec": round(timings["load_audio_sec"], 2),
            "pad_trim_sec": round(timings["pad_trim_sec"], 2),
            "mel_compute_sec": round(timings["mel_compute_sec"], 2),
            "decode_sec": round(timings["decode_sec"], 2),
            "total_pipeline_sec": round(total_time, 2),
        }
    }

    # -------------------------------------------------
    # Upload JSON to GCS
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

    log(f"Output Path            : gs://{GCS_OUTPUT_BUCKET}/{out_name}")
    log(f"Upload Time (s)        : {upload_time:.2f}")
    log(f"Total Pipeline (s)     : {total_time:.2f}")
    log("========== PIPELINE END ==========")

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

    name = urllib.parse.unquote(html.unescape(raw_name)) if raw_name else None
    file_size_mb = size_bytes / (1024 * 1024)

    log("========== EVENT RECEIVED ==========")
    log(f"Bucket                 : {bucket}")
    log(f"Raw Object Name        : {raw_name}")
    log(f"Decoded Object Name    : {name}")
    log(f"Generation             : {generation}")
    log(f"Reported Size (MB)     : {file_size_mb:.2f}")

    # -------------------------------------------------
    # Guards
    # -------------------------------------------------
    if not bucket or not name or not generation:
        log("[EVENT IGNORED] Missing required fields")
        return {"status": "ignored"}

    if bucket != GCS_INPUT_BUCKET:
        log("[EVENT IGNORED] Unexpected bucket")
        return {"status": "ignored"}

    if not name.lower().endswith((".mp3", ".wav", ".m4a", ".flac", ".ogg")):
        log("[EVENT IGNORED] Not an audio file")
        return {"status": "ignored"}

    # -------------------------------------------------
    # Download exact object generation
    # -------------------------------------------------
    with tempfile.NamedTemporaryFile(delete=False) as f:
        temp_audio = f.name

    blob = gcs_client.bucket(bucket).blob(name, generation=generation)

    try:
        t0 = time.time()
        blob.download_to_filename(temp_audio)
        log(f"Download Time (s)      : {time.time() - t0:.2f}")
    except NotFound:
        log(
            "[SKIPPED] Object no longer exists "
            f"(bucket={bucket}, name={name}, generation={generation})"
        )
        return {"status": "skipped", "reason": "object_not_found"}

    log(f"Temp File Path         : {temp_audio}")

    # -------------------------------------------------
    # Run pipeline
    # -------------------------------------------------
    try:
        run_whisper_pipeline(temp_audio, name, file_size_mb)
    finally:
        os.remove(temp_audio)
        log("Temp File Cleaned")

    log(
        f"[EVENT COMPLETE] Total Event Time (s): "
        f"{time.time() - event_start:.2f}"
    )

    return {"status": "processed", "file": name}
