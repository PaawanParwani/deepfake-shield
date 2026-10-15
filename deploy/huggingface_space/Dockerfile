FROM python:3.11-slim

# Install system dependencies for OpenCV and audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set up user for Hugging Face Spaces security
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR $HOME/app

# Copy dependency specifications and install
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code and models
COPY --chown=user:user . .

# Expose Hugging Face default port
EXPOSE 7860

# Launch production microservice
CMD ["uvicorn", "demo.backend.app:app", "--host", "0.0.0.0", "--port", "7860"]
