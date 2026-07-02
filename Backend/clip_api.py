import io
import re
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from flask import Flask, jsonify, request
from flask_cors import CORS

import clip

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR         = Path(__file__).parent
ANIMALS_TXT      = BASE_DIR / "animals_data.txt"
EMBED_CACHE      = BASE_DIR / "text_embeddings_cache.pt"
ANIMAL_THRESHOLD = 0.55    # raise to be stricter, lower to be more lenient

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[clip_api] Device: {DEVICE}")

app = Flask(__name__)
CORS(app)

# ──────────────────────────────────────────────────────────────
# NON-ANIMAL LABELS  (what to reject)
# ──────────────────────────────────────────────────────────────
NON_ANIMAL_LABELS = [
    "a photo of a person", "a photo of a human",
    "a photo of a man", "a photo of a woman", "a photo of a child",
    "a photo of a face",
    "a photo of a car", "a photo of a truck", "a photo of a bus",
    "a photo of a motorcycle", "a photo of a bicycle",
    "a photo of an airplane", "a photo of a boat", "a photo of a train",
    "a photo of a mobile phone", "a photo of a smartphone",
    "a photo of a computer", "a photo of a laptop",
    "a photo of a keyboard", "a photo of a television",
    "a photo of a monitor", "a photo of a camera",
    "a photo of a table", "a photo of a chair",
    "a photo of a sofa", "a photo of a bed",
    "a photo of a desk", "a photo of a shelf",
    "a photo of a bag", "a photo of a backpack",
    "a photo of a handbag", "a photo of shoes",
    "a photo of clothing", "a photo of a hat",
    "a photo of food", "a photo of a meal",
    "a photo of a drink", "a photo of a bottle", "a photo of a cup",
    "a photo of a tree", "a photo of a plant",
    "a photo of a flower", "a photo of grass", "a photo of a forest",
    "a photo of a building", "a photo of a house",
    "a photo of a road", "a photo of a street", "a photo of a room",
    "a photo of text", "a photo of a document",
    "a photo of a book", "a photo of a sign", "a photo of a poster",
    "a photo of a ball", "a photo of a toy",
    "a photo of a tool", "a photo of money",
]

ANIMAL_LABELS = [
    "a photo of an animal",
    "a photo of a wild animal",
    "a photo of a wildlife animal",
    "a photo of a pet",
    "a photo of a bird",
    "a photo of a mammal",
    "a photo of a reptile",
    "a photo of a fish",
    "a photo of an insect",
    "a photo of a creature",
]

# ──────────────────────────────────────────────────────────────
# LOAD CLIP (once at startup)
# ──────────────────────────────────────────────────────────────
print("[clip_api] Loading CLIP …")
clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False
print("[clip_api] CLIP loaded ✅")

def get_clip():
    """Return the loaded CLIP model and its preprocess transform."""
    return clip_model, clip_preprocess

# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def load_animals_txt(path: Path) -> List[str]:
    names = []
    if not path.exists():
        return names
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cleaned = re.sub(r"^\d+\.\s*", "", line).strip()
            if cleaned:
                names.append(cleaned)
    print(f"[clip_api] Loaded {len(names):,} names from animals_data.txt")
    return names


def build_text_embeddings(animal_names: List[str], cache_path: Path,
                          batch_size: int = 256) -> Tuple[torch.Tensor, List[str]]:
    if cache_path.exists():
        print(f"[clip_api] Loading cached embeddings …")
        data = torch.load(cache_path, map_location="cpu")
        return data["embeddings"], data["names"]

    print(f"[clip_api] Building text embeddings for {len(animal_names):,} names …")
    all_embeds = []
    for i in range(0, len(animal_names), batch_size):
        batch   = animal_names[i: i + batch_size]
        prompts = [f"a photo of a {n}" for n in batch]
        tokens  = clip.tokenize(prompts, truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = clip_model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
        all_embeds.append(emb.cpu().float())

    text_matrix = torch.cat(all_embeds, dim=0)
    torch.save({"embeddings": text_matrix, "names": animal_names}, cache_path)
    print(f"[clip_api] Cache saved ✅")
    return text_matrix, animal_names


def encode_image(pil_img: Image.Image) -> torch.Tensor:
    tensor = clip_preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = clip_model.encode_image(tensor)
        feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0).float().cpu()


def is_animal_image(img_embedding: torch.Tensor) -> Tuple[bool, float, str]:
    """
    Returns (is_animal, animal_score, rejection_reason).
    Uses CLIP to compare image against animal vs non-animal label sets.
    """
    all_labels = ANIMAL_LABELS + NON_ANIMAL_LABELS
    tokens     = clip.tokenize(all_labels, truncate=True).to(DEVICE)

    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = F.normalize(text_feats, dim=-1).float().cpu()

    sims  = img_embedding @ text_feats.T
    probs = F.softmax(sims * 100.0, dim=0)

    n_animal     = len(ANIMAL_LABELS)
    animal_score = probs[:n_animal].sum().item()

    # find the top non-animal label for the rejection message
    non_probs   = probs[n_animal:]
    top_non_idx = non_probs.argmax().item()
    top_non_lbl = (NON_ANIMAL_LABELS[top_non_idx]
                   .replace("a photo of ", "")
                   .replace("an ", "")
                   .strip())

    if animal_score >= ANIMAL_THRESHOLD:
        return True, animal_score, ""

    reason = f"Image appears to contain '{top_non_lbl}', not an animal."
    return False, animal_score, reason


def clip_predict_one(img_embedding: torch.Tensor,
                     text_matrix: torch.Tensor,
                     animal_names: List[str]) -> Tuple[str, float]:
    sims     = img_embedding @ text_matrix.T
    probs    = F.softmax(sims * 100.0, dim=0)
    best_idx = probs.argmax().item()
    return animal_names[best_idx], round(probs[best_idx].item() * 100.0, 2)


# ──────────────────────────────────────────────────────────────
# STARTUP  — load data once
# ──────────────────────────────────────────────────────────────
animal_names = load_animals_txt(ANIMALS_TXT)
text_matrix, animal_names = build_text_embeddings(animal_names, EMBED_CACHE)
print(f"[clip_api] Ready — {len(animal_names):,} species loaded ✅")


# ──────────────────────────────────────────────────────────────
# ENDPOINTS
# ──────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    """Root endpoint — confirms the server is running."""
    return jsonify({
        "status"  : "ok",
        "message" : "Animal Classification API is running",
        "endpoints": {
            "health" : "GET  /health",
            "predict": "POST /predict",
        },
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status"         : "ok",
        "animals_loaded" : len(animal_names),
        "device"         : DEVICE,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Request  : multipart/form-data  { image: <file> }

    Response (animal image):
        200 { "predicted_animal": "Snow Leopard", "accuracy": 87.35 }

    Response (non-animal image):
        400 { "error": "Not an animal image",
              "detail": "Image appears to contain 'car', not an animal.",
              "animal_score": 12.3 }

    Frontend fetch example:
    ───────────────────────
    const form = new FormData();
    form.append("image", file);
    const res  = await fetch("http://localhost:5000/predict", {
        method: "POST", body: form
    });
    const data = await res.json();

    if (!res.ok) {
        // non-animal image
        console.log(data.error);   // "Not an animal image"
        console.log(data.detail);  // "Image appears to contain 'car' …"
    } else {
        console.log(data.predicted_animal);  // "Snow Leopard"
        console.log(data.accuracy);          // 87.35
    }
    """
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img = Image.open(io.BytesIO(file.read())).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    try:
        img_embedding = encode_image(img)

        # ── GUARD: reject non-animal images
        is_animal, animal_score, reason = is_animal_image(img_embedding)

        if not is_animal:
            return jsonify({
                "error"        : "Not an animal image",
                "detail"       : reason,
                "animal_score" : round(animal_score * 100, 2),
            }), 400

        # ── PREDICT: only animal images reach here
        predicted_name, accuracy = clip_predict_one(
            img_embedding, text_matrix, animal_names
        )

        return jsonify({
            "predicted_animal" : predicted_name,
            "accuracy"         : accuracy,
        })

    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n[clip_api] Endpoints:")
    print("  GET  http://localhost:5000/health")
    print("  POST http://localhost:5000/predict\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
