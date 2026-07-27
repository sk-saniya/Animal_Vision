# ──────────────────────────────────────────────
# Dockerfile — Animal Vision Backend (Render)
# ──────────────────────────────────────────────

FROM python:3.10-slim

# System dependencies (git needed for CLIP install)
RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch first (avoids OOM during build)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all backend files
COPY . .

# Render uses the PORT environment variable
ENV PORT=5000
EXPOSE 5000

CMD ["python", "clip_api.py"]
