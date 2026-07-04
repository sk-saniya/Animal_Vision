FROM python:3.9-slim

# Install git for CLIP
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Pre-install CPU-only PyTorch to prevent build OOMs and speed up build
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and install the rest
COPY requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# Install flask, flask-cors, and openai-clip
RUN pip install --no-cache-dir flask flask-cors
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git

# Set up non-root user for Hugging Face Spaces (user ID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

# Copy application files (with ownership set to user)
COPY --chown=user . $HOME/app

EXPOSE 7860

CMD ["python", "clip_api.py"]
