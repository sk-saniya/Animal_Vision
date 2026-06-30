
# import sys
# from pathlib import Path
# from typing import List, Tuple

# import torch
# import torch.nn.functional as F
# from PIL import Image

# from clip_api import (
#     DEVICE,
#     build_text_embeddings,
#     encode_image,
#     load_animals_txt,
# )

# # ──────────────────────────────────────────────────────────────
# # CONFIG
# # ──────────────────────────────────────────────────────────────
# BASE_DIR    = Path(__file__).parent
# ANIMALS_TXT = BASE_DIR / "animals_data.txt"
# EMBED_CACHE = BASE_DIR / "text_embeddings_cache.pt"


# # ──────────────────────────────────────────────────────────────
# # CLIP zero-shot → ONE best name + accuracy
# # ──────────────────────────────────────────────────────────────
# def clip_predict_one(
#     img_embedding : torch.Tensor,   # (512,) CPU normalised
#     text_matrix   : torch.Tensor,   # (N, 512) all txt name embeddings
#     animal_names  : List[str],      # all names from animals_data.txt
# ) -> Tuple[str, float]:
   
#     # cosine similarity (both L2-normalised → dot product)
#     sims     = img_embedding @ text_matrix.T    # (N,)
#     probs    = F.softmax(sims * 100.0, dim=0)   # temperature=100 (CLIP paper)
#     best_idx = probs.argmax().item()

#     predicted_name = animal_names[best_idx]
#     accuracy       = probs[best_idx].item() * 100.0   # convert to percentage

#     return predicted_name, accuracy


# # ──────────────────────────────────────────────────────────────
# # MAIN
# # ──────────────────────────────────────────────────────────────
# def predict(image_path: str):

#     # load all names from animals_data.txt
#     animal_names = load_animals_txt(ANIMALS_TXT)

#     # build / load cached text embeddings for all names
#     text_matrix, animal_names = build_text_embeddings(
#         animal_names, cache_path=EMBED_CACHE
#     )

#     # encode input image via CLIP
#     img           = Image.open(image_path).convert("RGB")
#     img_embedding = encode_image(img)   # (512,) normalised CPU tensor

#     # CLIP zero-shot → ONE accurate name + accuracy
#     predicted_name, accuracy = clip_predict_one(
#         img_embedding, text_matrix, animal_names
#     )

#     # ── output
#     print(f"\n{'='*40}")
#     print(f"  Predicted Animal : {predicted_name}")
#     print(f"  Accuracy         : {accuracy:.2f}%")
#     print(f"{'='*40}\n")

#     return predicted_name, accuracy


# # ──────────────────────────────────────────────────────────────
# # ENTRY POINT
# # ──────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("Usage: python predict.py path/to/image.jpg")
#         sys.exit(0)

#     predict(sys.argv[1])

"""
predict.py
----------
Predict ONE accurate animal name with confidence score.
Uses CLIP zero-shot against ALL names in animals_data.txt.

GUARD:
  Before predicting, CLIP first checks if the image IS an animal.
  If it detects a non-animal object (human, car, bag, tree, phone,
  table, computer, etc.) it REJECTS the image with a clear message.
  Only animal images pass through to prediction.

Usage:
    python predict.py path/to/image.jpg

Files needed in same directory:
    animals_data.txt
    text_embeddings_cache.pt   (auto-built on first run)
"""

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image

import clip
from clip_api import (
    DEVICE,
    build_text_embeddings,
    encode_image,
    get_clip,
    load_animals_txt,
)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
ANIMALS_TXT     = BASE_DIR / "animals_data.txt"
EMBED_CACHE     = BASE_DIR / "text_embeddings_cache.pt"

# Confidence threshold — if CLIP's "animal" score is below this,
# the image is rejected as non-animal.
ANIMAL_THRESHOLD = 0.55   # 55 %  (tunable: raise to be stricter)

# ──────────────────────────────────────────────────────────────
# NON-ANIMAL CATEGORIES  (what to reject)
# ──────────────────────────────────────────────────────────────
NON_ANIMAL_LABELS = [
    # people
    "a photo of a person",
    "a photo of a human",
    "a photo of a man",
    "a photo of a woman",
    "a photo of a child",
    "a photo of a face",
    # vehicles
    "a photo of a car",
    "a photo of a truck",
    "a photo of a bus",
    "a photo of a motorcycle",
    "a photo of a bicycle",
    "a photo of an airplane",
    "a photo of a boat",
    "a photo of a train",
    # electronics
    "a photo of a mobile phone",
    "a photo of a smartphone",
    "a photo of a computer",
    "a photo of a laptop",
    "a photo of a keyboard",
    "a photo of a television",
    "a photo of a monitor",
    "a photo of a camera",
    # furniture / objects
    "a photo of a table",
    "a photo of a chair",
    "a photo of a sofa",
    "a photo of a bed",
    "a photo of a desk",
    "a photo of a shelf",
    # bags / clothing
    "a photo of a bag",
    "a photo of a backpack",
    "a photo of a handbag",
    "a photo of shoes",
    "a photo of clothing",
    "a photo of a hat",
    # food / drinks
    "a photo of food",
    "a photo of a meal",
    "a photo of a drink",
    "a photo of a bottle",
    "a photo of a cup",
    # plants / nature (non-animal)
    "a photo of a tree",
    "a photo of a plant",
    "a photo of a flower",
    "a photo of grass",
    "a photo of a forest",
    # buildings / places
    "a photo of a building",
    "a photo of a house",
    "a photo of a road",
    "a photo of a street",
    "a photo of a room",
    # text / documents
    "a photo of text",
    "a photo of a document",
    "a photo of a book",
    "a photo of a sign",
    "a photo of a poster",
    # sports / misc
    "a photo of a ball",
    "a photo of a toy",
    "a photo of a tool",
    "a photo of a weapon",
    "a photo of money",
]

# What an animal image looks like to CLIP
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
# GUARD — is this image an animal?
# ──────────────────────────────────────────────────────────────
def is_animal_image(img_embedding: torch.Tensor) -> Tuple[bool, float, str]:
    """
    Use CLIP to decide if the image is an animal or a non-animal object.

    Returns:
        is_animal   : True  → proceed with prediction
                      False → reject image
        confidence  : CLIP's animal score (0.0 – 1.0)
        reason      : human-readable rejection reason if not animal
    """
    clip_model, _ = get_clip()

    # encode BOTH animal labels and non-animal labels
    all_labels  = ANIMAL_LABELS + NON_ANIMAL_LABELS
    tokens      = clip.tokenize(all_labels, truncate=True).to(DEVICE)

    with torch.no_grad():
        text_feats = clip_model.encode_text(tokens)
        text_feats = F.normalize(text_feats, dim=-1).float().cpu()

    sims  = img_embedding @ text_feats.T        # (N,)
    probs = F.softmax(sims * 100.0, dim=0)      # (N,)

    n_animal     = len(ANIMAL_LABELS)
    animal_score = probs[:n_animal].sum().item()       # sum of all animal label probs
    non_animal_score = probs[n_animal:].sum().item()   # sum of all non-animal probs

    # find top non-animal label (for rejection message)
    non_probs   = probs[n_animal:]
    top_non_idx = non_probs.argmax().item()
    top_non_lbl = NON_ANIMAL_LABELS[top_non_idx].replace("a photo of ", "").replace("an ", "").strip()

    if animal_score >= ANIMAL_THRESHOLD:
        return True, animal_score, ""
    else:
        reason = (
            f"This image appears to contain {top_non_lbl}, not an animal.\n"
            f"  Please upload a photo of an animal."
        )
        return False, animal_score, reason


# ──────────────────────────────────────────────────────────────
# CLIP zero-shot → ONE best name + accuracy
# ──────────────────────────────────────────────────────────────
def clip_predict_one(
    img_embedding : torch.Tensor,
    text_matrix   : torch.Tensor,
    animal_names  : List[str],
) -> Tuple[str, float]:
    sims     = img_embedding @ text_matrix.T
    probs    = F.softmax(sims * 100.0, dim=0)
    best_idx = probs.argmax().item()
    return animal_names[best_idx], round(probs[best_idx].item() * 100.0, 2)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def predict(image_path: str):

    # load animals_data.txt + text embeddings
    animal_names = load_animals_txt(ANIMALS_TXT)
    text_matrix, animal_names = build_text_embeddings(
        animal_names, cache_path=EMBED_CACHE
    )

    # encode image
    img           = Image.open(image_path).convert("RGB")
    img_embedding = encode_image(img)

    # ── STEP 1: GUARD — reject non-animal images
    is_animal, animal_score, reason = is_animal_image(img_embedding)

    if not is_animal:
        print(f"\n{'='*45}")
        print(f"  ❌  NOT AN ANIMAL IMAGE")
        print(f"{'='*45}")
        print(f"  {reason}")
        print(f"{'='*45}\n")
        return None, 0.0

    # ── STEP 2: PREDICT — only animal images reach here
    predicted_name, accuracy = clip_predict_one(
        img_embedding, text_matrix, animal_names
    )

    print(f"\n{'='*45}")
    print(f"  Predicted Animal : {predicted_name}")
    print(f"  Accuracy         : {accuracy:.2f}%")
    print(f"{'='*45}\n")

    return predicted_name, accuracy


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py path/to/image.jpg")
        sys.exit(0)

    predict(sys.argv[1])