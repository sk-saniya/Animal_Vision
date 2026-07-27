import io
import os
import re
from pathlib import Path
from typing import List, Tuple
from PIL import Image

import torch
import torch.nn.functional as F
from ultralytics import YOLO
import clip

# ──────────────────────────────────────────────────────────────
# CONFIG & MODEL SETUP
# ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
ANIMALS_TXT = BASE_DIR / "animals_data.txt"
EMBED_CACHE = BASE_DIR / "text_embeddings_cache_v4.pt"
YOLO_MODEL = "yolov8x-oiv7.pt"

COLORS = [
    "#FF4444", "#44BB44", "#4488FF", "#FFD700",
    "#FF44FF", "#00CCCC", "#FF8C00", "#AA44FF",
    "#FF6B6B", "#44FF88", "#FF44AA", "#88FFFF",
    "#FFAA44", "#44AAFF", "#FF44CC", "#CCFF44",
]
YOLO_MULTI_IMGSZ = [960, 1280, 1536]

ANIMAL_THRESHOLD = 0.55
CLIP_LOGIT_SCALE = 45.0       # Raised: sharpens softmax, reduces obscure-species noise
YOLO_LABEL_BOOST = 0.10       # Raised: stronger push toward YOLO-detected family
YOLO_HARD_OVERRIDE_CONF = 0.60  # If YOLO is this confident, prefer its genus family
YOLO_ANIMAL_OVERRIDE_CONF = 0.14
SAME_OBJECT_IOU = 0.55
MAX_BOX_AREA_RATIO = 0.85
MIN_BOX_AREA_RATIO = 0.0004

# Canonical genus labels that map to YOLO class families.
# When YOLO is very confident, we restrict CLIP search to these family terms.
YOLO_FAMILY_KEYWORDS = {
    "cat":       ["cat", "domestic cat", "kitten", "tabby", "feline"],
    "dog":       ["dog", "domestic dog", "puppy", "canine", "hound"],
    "bird":      ["bird", "parrot", "eagle", "owl", "hawk", "sparrow", "robin"],
    "fish":      ["fish", "salmon", "tuna", "goldfish", "clownfish"],
    "horse":     ["horse", "pony", "stallion", "mare", "foal"],
    "cow":       ["cow", "bull", "calf", "bovine", "cattle"],
    "sheep":     ["sheep", "lamb", "ewe", "ram"],
    "rabbit":    ["rabbit", "hare", "bunny"],
    "elephant":  ["elephant"],
    "bear":      ["bear", "polar bear", "grizzly bear", "brown bear"],
    "tiger":     ["tiger"],
    "lion":      ["lion"],
    "cheetah":   ["cheetah"],
    "leopard":   ["leopard", "jaguar", "panther"],
    "zebra":     ["zebra"],
    "giraffe":   ["giraffe"],
    "monkey":    ["monkey", "ape", "chimpanzee", "gorilla", "baboon"],
    "snake":     ["snake", "cobra", "viper", "python"],
    "crocodile": ["crocodile", "alligator"],
    "turtle":    ["turtle", "tortoise"],
    "penguin":   ["penguin"],
    "shark":     ["shark"],
    "duck":      ["duck", "goose", "swan"],
    "deer":      ["deer", "elk", "moose", "reindeer"],
    "wolf":      ["wolf"],
    "fox":       ["fox"],
}

PROMPT_TEMPLATES = [
    "a clear photo of a {}",
    "a close-up photo of a {}",
    "a wildlife photo of a {}",
    "a photo of the animal {}",
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

YOLO_CLASS_TO_LABEL = {
    "antelope": "antelope", "badger": "badger", "bat": "bat", "bear": "bear",
    "bee": "bee", "beetle": "beetle", "bison": "bison", "boar": "boar",
    "buffalo": "buffalo", "butterfly": "butterfly", "camel": "camel",
    "cat": "domestic cat", "cattle": "cow", "cheetah": "cheetah",
    "chicken": "chicken", "cow": "cow", "crocodile": "crocodile",
    "deer": "deer", "dog": "domestic dog", "duck": "duck", "eagle": "eagle",
    "elephant": "elephant", "fish": "fish", "fox": "fox", "frog": "frog",
    "giraffe": "giraffe", "goat": "goat", "goose": "goose", "horse": "horse",
    "kangaroo": "kangaroo", "leopard": "leopard", "lion": "lion",
    "lizard": "lizard", "monkey": "monkey", "mouse": "mouse", "otter": "otter",
    "owl": "owl", "panda": "panda", "parrot": "parrot", "penguin": "penguin",
    "pig": "pig", "rabbit": "rabbit", "rhinoceros": "rhinoceros",
    "seal": "seal", "shark": "shark", "sheep": "sheep", "snake": "snake",
    "squirrel": "squirrel", "tiger": "tiger", "turtle": "turtle",
    "whale": "whale", "zebra": "zebra",
}

NON_ANIMAL_LABELS = [
    "a photo of a person", "a photo of a human", "a photo of a man", "a photo of a woman", "a photo of a child",
    "a photo of a face", "a photo of a car", "a photo of a truck", "a photo of a bus", "a photo of a motorcycle",
    "a photo of a bicycle", "a photo of an airplane", "a photo of a boat", "a photo of a train",
    "a photo of a mobile phone", "a photo of a smartphone", "a photo of a computer", "a photo of a laptop",
    "a photo of a keyboard", "a photo of a television", "a photo of a monitor", "a photo of a camera",
    "a photo of a table", "a photo of a chair", "a photo of a sofa", "a photo of a bed",
    "a photo of a desk", "a photo of a shelf", "a photo of a bag", "a photo of a backpack",
    "a photo of a handbag", "a photo of shoes", "a photo of clothing", "a photo of a hat",
    "a photo of food", "a photo of a meal", "a photo of a drink", "a photo of a bottle", "a photo of a cup",
    "a photo of a tree", "a photo of a plant", "a photo of a flower", "a photo of grass", "a photo of a forest",
    "a photo of a building", "a photo of a house", "a photo of a road", "a photo of a street", "a photo of a room",
    "a photo of text", "a photo of a document", "a photo of a book", "a photo of a sign", "a photo of a poster",
    "a photo of a ball", "a photo of a toy", "a photo of a tool", "a photo of money",
]

ANIMAL_LABELS = [
    "a photo of an animal", "a photo of a wild animal", "a photo of a wildlife animal",
    "a photo of a pet", "a photo of a bird", "a photo of a mammal", "a photo of a reptile",
    "a photo of a fish", "a photo of an insect", "a photo of a creature",
]

# Lazy-loaded globals
clip_model = None
clip_preprocess = None
yolo_model = None
animal_names = []
text_matrix = None

def get_clip():
    global clip_model, clip_preprocess
    if clip_model is None:
        print("[predict_single] Loading CLIP model (ViT-B/32)...")
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=DEVICE)
        clip_model.eval()
        for p in clip_model.parameters():
            p.requires_grad = False
    return clip_model, clip_preprocess

def get_yolo():
    global yolo_model
    if yolo_model is None:
        possible_paths = [
            Path(YOLO_MODEL),
            BASE_DIR / YOLO_MODEL,
            BASE_DIR.parent / "Backend" / YOLO_MODEL,
            Path("yolov8n.pt"),
            BASE_DIR / "yolov8n.pt",
        ]
        weights_path = YOLO_MODEL
        for p in possible_paths:
            if p.exists():
                weights_path = str(p)
                break
        print(f"[predict_single] Loading YOLOv8 from {weights_path}...")
        yolo_model = YOLO(weights_path)
    return yolo_model

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
    return names

def build_text_embeddings(names: List[str], cache_path: Path, batch_size: int = 64) -> Tuple[torch.Tensor, List[str]]:
    model, _ = get_clip()
    if cache_path.exists():
        data = torch.load(cache_path, map_location="cpu")
        if data.get("templates") == PROMPT_TEMPLATES:
            return data["embeddings"], data["names"]
    
    print(f"[predict_single] Building text embeddings for {len(names):,} species...")
    all_embeds = []
    total = len(names)
    for i in range(0, total, batch_size):
        batch = names[i: i + batch_size]
        prompts = [template.format(n.lower()) for n in batch for template in PROMPT_TEMPLATES]
        tokens = clip.tokenize(prompts, truncate=True).to(DEVICE)
        with torch.no_grad():
            emb = model.encode_text(tokens)
            emb = F.normalize(emb, dim=-1)
            emb = emb.view(len(batch), len(PROMPT_TEMPLATES), -1).mean(dim=1)
            emb = F.normalize(emb, dim=-1)
        all_embeds.append(emb.cpu().float())

    matrix = torch.cat(all_embeds, dim=0)
    torch.save({"embeddings": matrix, "names": names, "templates": PROMPT_TEMPLATES}, cache_path)
    return matrix, names

def ensure_initialized():
    global animal_names, text_matrix
    if text_matrix is None or not animal_names:
        get_clip()
        animal_names = load_animals_txt(ANIMALS_TXT)
        text_matrix, animal_names = build_text_embeddings(animal_names, EMBED_CACHE)

def encode_image(pil_img: Image.Image) -> torch.Tensor:
    model, preprocess = get_clip()
    tensor = preprocess(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = model.encode_image(tensor)
        feat = F.normalize(feat, dim=-1)
    return feat.squeeze(0).float().cpu()

def is_animal_image(img_embedding: torch.Tensor) -> Tuple[bool, float, str]:
    model, _ = get_clip()
    all_labels = ANIMAL_LABELS + NON_ANIMAL_LABELS
    tokens = clip.tokenize(all_labels, truncate=True).to(DEVICE)
    with torch.no_grad():
        text_feats = model.encode_text(tokens)
        text_feats = F.normalize(text_feats, dim=-1).float().cpu()
    sims = img_embedding @ text_feats.T
    probs = F.softmax(sims * 100.0, dim=0)

    n_animal = len(ANIMAL_LABELS)
    animal_score = probs[:n_animal].sum().item()

    non_probs = probs[n_animal:]
    top_non_idx = non_probs.argmax().item()
    top_non_lbl = NON_ANIMAL_LABELS[top_non_idx].replace("a photo of ", "").replace("an ", "").strip()

    if animal_score >= ANIMAL_THRESHOLD:
        return True, animal_score, ""
    reason = f"Image appears to contain '{top_non_lbl}', not an animal."
    return False, animal_score, reason

def _box_area(box: list) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])

def _compute_iou(box_a: list, box_b: list) -> float:
    x1, y1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
    x2, y2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = _box_area(box_a)
    area_b = _box_area(box_b)
    return inter / (area_a + area_b - inter)

def detect_animal_objects(img: Image.Image) -> List[dict]:
    W, H = img.size
    img_area = W * H
    yolo = get_yolo()
    results = yolo(img, conf=0.10, iou=0.45, imgsz=1280, augment=True, verbose=False)
    result = results[0]

    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    confidences = result.boxes.conf.cpu().numpy()
    xyxy = result.boxes.xyxy.cpu().numpy().astype(int)
    class_names = result.names

    detections = []
    for cls_id, conf_score, box in zip(class_ids, confidences, xyxy):
        cname = class_names.get(int(cls_id), str(cls_id))
        if not any(kw in cname.lower() for kw in ANIMAL_KEYWORDS):
            continue
        x1, y1, x2, y2 = box.tolist()
        clipped_box = [max(0, x1), max(0, y1), min(W, x2), min(H, y2)]
        b_area = _box_area(clipped_box)
        if b_area > MAX_BOX_AREA_RATIO * img_area or b_area < MIN_BOX_AREA_RATIO * img_area:
            continue
        detections.append({"box": clipped_box, "yolo_class": cname, "confidence": float(conf_score)})
    return detections

def crop_with_padding(original_img: Image.Image, box: list, padding_ratio: float = 0.12) -> Image.Image:
    W, H = original_img.size
    x1, y1, x2, y2 = box
    pad_x = max(int((x2 - x1) * padding_ratio), 8)
    pad_y = max(int((y2 - y1) * padding_ratio), 8)
    return original_img.crop((max(0, x1 - pad_x), max(0, y1 - pad_y), min(W, x2 + pad_x), min(H, y2 + pad_y)))

def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def clip_predict_one(
    img_embedding: torch.Tensor,
    yolo_class: str | None = None,
    yolo_confidence: float = 0.0,
    context_embedding: torch.Tensor | None = None,
) -> Tuple[str, float]:
    """CLIP zero-shot prediction with three accuracy improvements:

    1. YOLO class boosting  — nudge similarity scores toward species matching
       what YOLO detected (e.g. YOLO says 'cat' → boost all cat/feline names).
    2. Crop + context ensemble — if a context (padded) embedding is supplied,
       average it with the tight crop embedding before ranking.
    3. YOLO hard override  — if YOLO confidence is very high AND CLIP's top
       answer belongs to a different animal family, trust YOLO's family.
    """
    ensure_initialized()

    # ── 1. Ensemble: average crop + context embeddings
    if context_embedding is not None:
        combined = F.normalize(
            (img_embedding + context_embedding) * 0.5, dim=-1
        )
    else:
        combined = img_embedding

    sims = combined @ text_matrix.T  # (N,)

    # ── 2. YOLO class boost: add a small score to species that match the
    #       YOLO-detected genus (e.g. if YOLO says 'cat', boost all cat names)
    yolo_key = _normalize_label(yolo_class or "")
    family_terms = None
    for key, terms in YOLO_FAMILY_KEYWORDS.items():
        if key in yolo_key or yolo_key in key:
            family_terms = terms
            break
    # Also check YOLO_CLASS_TO_LABEL mapping
    canonical = YOLO_CLASS_TO_LABEL.get(yolo_key)
    if canonical:
        extra_terms = [_normalize_label(canonical)]
        family_terms = list(dict.fromkeys((family_terms or []) + extra_terms))

    if family_terms:
        boosts = torch.zeros(len(animal_names))
        for idx, name in enumerate(animal_names):
            normalized = _normalize_label(name)
            if any(term in normalized for term in family_terms):
                boosts[idx] = YOLO_LABEL_BOOST
        sims = sims + boosts

    probs = F.softmax(sims * CLIP_LOGIT_SCALE, dim=0)
    best_idx = int(probs.argmax().item())
    best_name = animal_names[best_idx]
    best_prob = probs[best_idx].item()
    best_confidence = round(best_prob * 100.0, 2)

    # ── 3. YOLO hard override: if YOLO is very confident but CLIP landed in a
    #       completely different family, find the highest-scoring name in
    #       YOLO's family and return that instead.
    if (
        family_terms
        and yolo_confidence >= YOLO_HARD_OVERRIDE_CONF
        and not any(term in _normalize_label(best_name) for term in family_terms)
    ):
        # Collect (prob, name) for all names in YOLO's family
        family_candidates = [
            (probs[i].item(), animal_names[i])
            for i, name in enumerate(animal_names)
            if any(term in _normalize_label(name) for term in family_terms)
        ]
        if family_candidates:
            family_prob, family_name = max(family_candidates, key=lambda x: x[0])
            # Only override if family's best is at least reasonably scored
            if family_prob > 0.0001:
                return family_name, round(family_prob * 100.0, 2)

    return best_name, best_confidence

def handle_predict_single(img: Image.Image) -> Tuple[dict, int]:
    """
    Main handler for single-animal prediction endpoint.
    """
    ensure_initialized()
    img_embedding = encode_image(img)
    is_anim, animal_score, reason = is_animal_image(img_embedding)

    if not is_anim:
        return {
            "error": "Not an animal image",
            "detail": reason,
            "animal_score": round(animal_score * 100, 2),
        }, 400

    kept_detections = detect_animal_objects(img)
    if len(kept_detections) > 1:
        return {
            "error": "Multiple animals detected",
            "detail": "This image contains multiple animals. Please use the Multiple Animal Image section.",
        }, 400

    classify_img = img
    context_img = None
    yolo_class = None
    yolo_confidence = 0.0
    if len(kept_detections) == 1:
        det = kept_detections[0]
        yolo_class = det.get("yolo_class")
        yolo_confidence = det.get("confidence", 0.0)
        # Tight crop focused on the animal
        classify_img = crop_with_padding(img, det["box"], padding_ratio=0.05)
        # Wider context crop: more background context helps CLIP distinguish species
        context_img = crop_with_padding(img, det["box"], padding_ratio=0.30)

    classify_embedding = encode_image(classify_img)
    context_embedding = encode_image(context_img) if context_img is not None else None
    predicted_name, accuracy = clip_predict_one(
        classify_embedding,
        yolo_class=yolo_class,
        yolo_confidence=yolo_confidence,
        context_embedding=context_embedding,
    )

    return {
        "predicted_animal": predicted_name,
        "accuracy": accuracy,
    }, 200

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        test_img = Image.open(sys.argv[1]).convert("RGB")
        res, code = handle_predict_single(test_img)
        print("Result:", res, f"({code})")
