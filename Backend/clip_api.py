
import io
import os
import re
from pathlib import Path
from typing import List, Tuple
import base64
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from flask import Flask, jsonify, request
from flask_cors import CORS
from ultralytics import YOLO

import clip

# ──────────────────────────────────────────────────────────────
# MULTIPLE ANIMAL DETECTION & CLASSIFICATION CONFIG
# ──────────────────────────────────────────────────────────────
COLORS = [
    "#FF4444", "#44BB44", "#4488FF", "#FFD700",
    "#FF44FF", "#00CCCC", "#FF8C00", "#AA44FF",
    "#FF6B6B", "#44FF88", "#FF44AA", "#88FFFF",
    "#FFAA44", "#44AAFF", "#FF44CC", "#CCFF44",
]

YOLO_MODEL = "yolov8x-oiv7.pt"
YOLO_CONF = 0.10
YOLO_IOU = 0.45
YOLO_IMGSZ = 1280
YOLO_AUGMENT = True
YOLO_AGNOSTIC_NMS = False
SAME_OBJECT_IOU = 0.55

ANIMAL_KEYWORDS = {
    "animal", "mammal", "bird", "reptile", "fish", "insect", "amphibian",
    "cat", "dog", "rabbit", "horse", "cow", "sheep", "goat", "pig",
    "lion", "tiger", "bear", "elephant", "giraffe", "zebra", "monkey",
    "gorilla", "chimpanzee", "deer", "fox", "wolf", "leopard", "cheetah",
    "jaguar", "panda", "koala", "kangaroo", "crocodile", "alligator",
    "turtle", "snake", "lizard", "frog", "parrot", "eagle", "owl",
    "penguin", "duck", "goose", "swan", "flamingo", "peacock", "chicken",
    "turkey", "shark", "dolphin", "whale", "seal", "otter", "squirrel",
    "hamster", "mouse", "rat", "hedgehog", "bat", "butterfly", "bee",
    "ant", "spider", "crab", "lobster", "jellyfish", "octopus", "goldfish",
    "macaw", "cockatoo", "guinea pig", "ferret", "raccoon", "skunk",
    "camel", "llama", "alpaca", "bison", "buffalo", "moose", "elk",
    "reindeer", "hippopotamus", "rhinoceros", "meerkat", "hyena",
    "bobcat", "lynx", "cougar", "panther", "ocelot", "platypus",
    "wombat", "possum", "wallaby", "gibbon", "orangutan", "baboon",
    "lemur", "iguana", "gecko", "chameleon", "komodo", "python",
    "cobra", "viper", "salmon", "tuna", "clownfish", "sparrow", "robin",
    "cardinal", "finch", "hummingbird", "pigeon", "dove", "crow", "raven",
    "woodpecker", "toucan", "pelican", "heron", "stork", "vulture",
    "hawk", "falcon", "ostrich", "emu", "puffin", "albatross", "wildlife",
    "mule", "donkey", "pony", "bull", "ox", "calf", "kitten", "puppy", "carnivore",
    "sea lion", "walrus", "harbor seal", "sea otter", "polar bear", "orca",
    "narwhal", "sea turtle", "tortoise", "starfish", "seahorse", "stingray",
    "swordfish", "catfish", "koi", "eel", "worm", "earthworm", "snail", "slug",
    "centipede", "millipede", "scorpion", "tick", "shrimp", "oyster", "clam",
    "mussel", "coral", "caterpillar", "moth", "dragonfly", "grasshopper",
    "cricket", "cockroach", "termite", "ladybug", "beetle", "mosquito", "fly",
    "invertebrate", "vertebrate", "marine mammal", "canary", "magpie",
    "kingfisher", "seagull", "gull", "tern", "quail", "pheasant", "partridge",
    "armadillo", "porcupine", "mole", "shrew", "weasel", "mink", "badger",
    "tapir", "dugong", "manatee", "sea cucumber", "sea urchin", "tadpole",
    "newt", "salamander", "toad", "axolotl",
}

GENERIC_SUPERCLASSES = {
    "animal", "mammal", "bird", "reptile", "fish", "insect", "amphibian",
    "invertebrate", "vertebrate", "carnivore", "wildlife", "marine mammal",
    "marine invertebrate",
}

def _compute_iou(box_a: list, box_b: list) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)

def _box_area(box: list) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])

def _is_animal_class(class_name: str) -> bool:
    name = class_name.lower()
    return any(kw in name for kw in ANIMAL_KEYWORDS)

def _is_generic_yolo_class(class_name: str) -> bool:
    return class_name.lower().strip() in GENERIC_SUPERCLASSES

def _dedupe_same_object(detections: List[dict], iou_thresh: float = SAME_OBJECT_IOU) -> List[dict]:
    ordered = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    for det in ordered:
        merged = False
        for i, existing in enumerate(kept):
            if _compute_iou(det["box"], existing["box"]) < iou_thresh:
                continue

            if _is_generic_yolo_class(existing["yolo_class"]) and not _is_generic_yolo_class(det["yolo_class"]):
                kept[i] = det
            merged = True
            break

        if not merged:
            kept.append(det)

    return kept

def _remove_overarching_boxes(detections: List[dict]) -> List[dict]:
    filtered = []
    for i, det_a in enumerate(detections):
        x1a, y1a, x2a, y2a = det_a["box"]
        area_a = _box_area(det_a["box"])
        is_overarching = False

        for j, det_b in enumerate(detections):
            if i == j:
                continue

            area_b = _box_area(det_b["box"])
            x1b, y1b, x2b, y2b = det_b["box"]
            x_left = max(x1a, x1b)
            y_top = max(y1a, y1b)
            x_right = min(x2a, x2b)
            y_bottom = min(y2a, y2b)

            if x_right <= x_left or y_bottom <= y_top:
                continue

            intersection = (x_right - x_left) * (y_bottom - y_top)
            if (
                intersection > 0.92 * area_b
                and area_a > 3.0 * area_b
                and det_a["confidence"] < det_b["confidence"] - 0.10
            ):
                is_overarching = True
                break

        if not is_overarching:
            filtered.append(det_a)

    return filtered

def _detect_animal_objects(original_img: Image.Image) -> List[dict]:
    W, H = original_img.size
    yolo = get_yolo()
    results = yolo(
        original_img,
        conf=YOLO_CONF,
        iou=YOLO_IOU,
        imgsz=YOLO_IMGSZ,
        augment=YOLO_AUGMENT,
        agnostic_nms=YOLO_AGNOSTIC_NMS,
        verbose=False,
    )

    result = results[0]
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    class_names = result.names

    detections = []
    for cls_id, conf_score, box in zip(class_ids, confidences, xyxy):
        cname = class_names.get(int(cls_id), str(cls_id))
        if not _is_animal_class(cname):
            continue

        x1, y1, x2, y2 = box.tolist()
        detections.append({
            "box": [max(0, x1), max(0, y1), min(W, x2), min(H, y2)],
            "yolo_class": cname,
            "confidence": float(conf_score),
        })

    detections = _dedupe_same_object(detections)
    detections = _remove_overarching_boxes(detections)

    for i, det in enumerate(detections, start=1):
        det["label"] = i

    return detections

def _get_font(size: int = 22):
    for path in [
        "arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()

def _text_size(draw, text: str, font, fallback_char_w=12, fallback_h=24):
    try:
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return r - l, b - t
    except Exception:
        return len(text) * fallback_char_w, fallback_h

yolo_model = None
def get_yolo():
    global yolo_model
    if yolo_model is None:
        possible_paths = [
            Path(YOLO_MODEL),
            Path(__file__).parent / YOLO_MODEL,
            Path(__file__).parent.parent / "Backend" / YOLO_MODEL,
            Path("yolov8n.pt"),
            Path(__file__).parent / "yolov8n.pt",
            Path(__file__).parent.parent / "Backend" / "yolov8n.pt"
        ]
        weights_path = YOLO_MODEL
        for p in possible_paths:
            if p.exists():
                weights_path = str(p)
                break
        print(f"[clip_api] Loading YOLOv8 from {weights_path}...")
        yolo_model = YOLO(weights_path)
    return yolo_model


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



@app.route("/predict-multiple", methods=["POST"])
def predict_multiple():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        import io
        img_bytes = file.read()
        original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {str(e)}"}), 400

    try:
        W, H = original_img.size
        kept_detections = _detect_animal_objects(original_img)

        if not kept_detections:
            return jsonify({
                "error": "No animals detected",
                "detail": "Object detection did not find any animal objects in this image.",
            }), 400

        # Classify each detected animal crop using CLIP
        font = _get_font(20)
        final_detections = []
        annotated_img = original_img.copy()
        draw = ImageDraw.Draw(annotated_img)

        for i, det in enumerate(kept_detections, start=1):
            x1, y1, x2, y2 = det["box"]
            color = COLORS[(i - 1) % len(COLORS)]

            # Crop with padding
            pad_x = max(int((x2 - x1) * 0.05), 6)
            pad_y = max(int((y2 - y1) * 0.05), 6)
            x1c, y1c = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2c, y2c = min(W, x2 + pad_x), min(H, y2 + pad_y)
            crop = original_img.crop((x1c, y1c, x2c, y2c))

            crop_embedding = encode_image(crop)
            is_anim, animal_score, reason = is_animal_image(crop_embedding)

            if not is_anim:
                name, accuracy = None, 0.0
            else:
                name, accuracy = clip_predict_one(crop_embedding, text_matrix, animal_names)

            # Convert crop to base64
            crop_buffer = io.BytesIO()
            crop.save(crop_buffer, format="JPEG", quality=90)
            crop_base64 = base64.b64encode(crop_buffer.getvalue()).decode("utf-8")

            # Draw box on annotated image
            label_name = name if name else "Rejected"
            display_text = f"#{i} {label_name} ({accuracy:.1f}%)" if name else f"#{i} Not Animal"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            tw, th = _text_size(draw, display_text, font)
            ly = y1 - th - 8
            if ly < 0:
                ly = y1 + 4
            draw.rectangle([x1, ly, x1 + tw + 10, ly + th + 6], fill=color)
            draw.text((x1 + 5, ly + 3), display_text, fill="white", font=font)

            final_detections.append({
                "label": i,
                "box": det["box"],
                "name": label_name,
                "accuracy": accuracy,
                "is_animal": is_anim,
                "crop_image": f"data:image/jpeg;base64,{crop_base64}"
            })

        # Convert annotated image to base64
        ann_buffer = io.BytesIO()
        annotated_img.save(ann_buffer, format="JPEG", quality=90)
        ann_base64 = base64.b64encode(ann_buffer.getvalue()).decode("utf-8")

        return jsonify({
            "annotated_image": f"data:image/jpeg;base64,{ann_base64}",
            "detections": final_detections
        })

    except Exception as e:
        return jsonify({"error": f"Detection/Prediction failed: {str(e)}"}), 500


# ──────────────────────────────────────────────────────────────
# RUN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("\n[clip_api] Endpoints:")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  POST http://localhost:{port}/predict\n")
    app.run(host="0.0.0.0", port=port, debug=False)
