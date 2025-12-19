# Use an official Python runtime as a parent image
# FROM python:3.10
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory in the container
WORKDIR /app
# Install system deps INCLUDING Rust (key fix)
RUN apt-get update && apt-get install -y \
    build-essential \
    rustc \
    cargo \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Verify Rust (important)
RUN rustc --version && cargo --version

# Copy the requirements file into the container
COPY requirements.txt .

# Install the required packages
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

RUN mkdir models

# Download the Whisper model during the build
ENV MODEL_VERSION=large-v3-turbo
RUN mkdir -p /app/models && \
    python -c "import whisper; whisper.load_model('$MODEL_VERSION', download_root='/app/models')"

# Install ffmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the application code into the container
COPY main.py .

# Expose port 8080
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
