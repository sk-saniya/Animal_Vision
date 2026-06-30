# Animal Vision

A **zero-shot animal classification** web application powered by OpenAI's **CLIP** model. It can identify any of ~25,000 animal species from a single image without any model fine-tuning.


## Features

- **Zero-shot inference** using CLIP (ViT-B/32) – no training required.
- **25,000+ animal classes** loaded from `animals_data.txt`.
- **Top‑k predictions** (configurable, default 5) with confidence scores.
- **Responsive React UI** with drag‑and‑drop image upload, realtime predictions, and animated result display.
- **Backend API** built with Flask, exposing `/predict` and `/health` endpoints.
- **Portable** – runs on CPU or CUDA‑enabled GPU automatically.

---

## Architecture

```
Animal Vision
├─ Backend (Python / Flask)
│   ├─ clip_api.py      # CLIP model loading, inference, API routes
│   ├─ predict.py       # CLI helper for quick testing
│   └─ animals_data.txt # List of animal names (~25k)
└─ Frontend (Next.js / React)
    ├─ pages/api/predict.js   # Proxy to Flask `/predict`
    ├─ pages/api/health.js    # Proxy to Flask `/health`
    ├─ components/*           # UI components (UploadZone, PredictionResult, …)
    └─ pages/index.js         # Main page UI
```

---

## Installation

### Prerequisites

- **Python 3.9+** (virtual environment recommended)
- **Node.js 18+** and **npm**
- **Git** (to clone the repository)
- Optional: **CUDA‑enabled GPU** for faster inference

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/your-username/animal_vision.git
cd animal_vision/Backend

# Create and activate a virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# The first run will build a text‑embedding cache (~2 GB). This may take a few minutes.
python predict.py path/to/sample.jpg  # sanity‑check
```

### Frontend Setup

```bash
cd ../Frontend
npm install   # installs next, react, form-data, node-fetch, etc.

# Create a .env.local file (copy from .env.example if present) and set the Flask URL if different:
# FLASK_URL=http://localhost:5000

npm run dev   # launches the Next.js dev server at http://localhost:3000
```

---

## Running the Application

1. **Start the Flask backend** (ensure the virtual environment is active):
   ```bash
   python clip_api.py
   ```
   The server listens on `http://0.0.0.0:5000`.

2. **Start the Next.js frontend** (in a separate terminal):
   ```bash
   npm run dev
   ```
   Open the browser at `http://localhost:3000`.

3. Upload an image – the UI will display the predicted animal, confidence, and the top‑k alternatives.

---

## API Reference

### `GET /health`
Returns basic health information.
```json
{
  "status": "ok",
  "animals_loaded": 25000,
  "device": "cpu" // or "cuda"
}
```

### `POST /predict`
**Form data**:
- `image` – image file (multipart/form‑data)
- `top_k` (optional) – number of predictions to return (1‑len(animal_names))

**Response** (JSON):
```json
{
  "predicted_animal": "Tiger",
  "accuracy": 12.34,
  "predictions": [
    {"label": "Tiger", "confidence": 12.34},
    {"label": "Lion", "confidence": 9.87},
    ...
  ]
}
```

---

## Data & Model

- **Model**: OpenAI CLIP ViT‑B/32 (downloaded automatically on first run).
- **Classes**: `animals_data.txt` – each line contains a plain English animal name (e.g., `African Elephant`).
- **Text embeddings** are cached in `text_embeddings_cache.pt` for fast startup.

---

## Customization & Extending

- **Add new classes** – edit `animals_data.txt` and delete `text_embeddings_cache.pt`; the next start will rebuild the cache.
- **Change the prompt template** – modify `clip_api.py` where `prompts = [f"a photo of a {n}" for n in batch]`.
- **Switch to a different CLIP variant** – change the model string in `clip.load("ViT-B/32", …)`.
- **Expose additional endpoints** – follow the Flask pattern used for `/predict`.

---

## Contributing

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/awesome-thing`).
3. Make your changes, ensure all tests (if any) pass.
4. Submit a Pull Request with a clear description of the change.

Please adhere to the existing code style and include updated documentation when applicable.

---

## License

This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

*Happy animal hunting!*
