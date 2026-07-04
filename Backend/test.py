
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from clip_api import (
    DEVICE,
    build_text_embeddings,
    combined_predict,
    encode_image,
    get_clip,
    load_animals_txt,
    zero_shot_predict,
)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
DATASET_DIR = BASE_DIR / "./Animal"
ANIMALS_TXT = BASE_DIR / "animals_data.txt"
MODEL_PATH  = BASE_DIR / "cnn_model.pth"
LABEL_MAP   = BASE_DIR / "label_map.json"
EMBED_CACHE = BASE_DIR / "text_embeddings_cache.pt"

BATCH_SIZE  = 32
VAL_SPLIT   = 0.15
SEED        = 42
CLIP_DIM    = 512
TOP_K_CLIP  = 10     # how many CLIP candidates to pull from 25,000 names

print(f"[test] Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# MODEL (same architecture as train.py)
# ──────────────────────────────────────────────────────────────
class AnimalClassifier(nn.Module):
    def __init__(self, in_dim: int, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class AnimalImageDataset(Dataset):
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, class_to_idx: dict):
        self.class_to_idx = class_to_idx
        self.samples = []
        for cls, idx in class_to_idx.items():
            cls_dir = root / cls
            if not cls_dir.is_dir():
                continue
            for f in cls_dir.iterdir():
                if f.suffix.lower() in self.EXTS:
                    self.samples.append((f, idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img  = Image.open(path).convert("RGB")
        feat = encode_image(img)
        return feat, label, str(path)


# ──────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────
def softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ── Load label map
    with open(LABEL_MAP) as f:
        idx_to_class = {int(k): v for k, v in json.load(f).items()}
    class_to_idx = {v: k for k, v in idx_to_class.items()}
    num_classes  = len(class_to_idx)

    # ── Load animals_data.txt + text embeddings
    print("[test] Loading animals_data.txt and text embedding cache …")
    animal_names = load_animals_txt(ANIMALS_TXT)
    text_matrix, animal_names = build_text_embeddings(
        animal_names, cache_path=EMBED_CACHE
    )
    print(f"[test] Text matrix shape: {text_matrix.shape}")   # (25000+, 512)

    # ── Load dataset (same 15% split as training)
    full_ds = AnimalImageDataset(DATASET_DIR, class_to_idx)
    n_total = len(full_ds)
    n_test  = int(n_total * VAL_SPLIT)
    n_train = n_total - n_test
    _, test_ds = random_split(
        full_ds, [n_train, n_test],
        generator=torch.Generator().manual_seed(SEED)
    )
    test_loader = DataLoader(test_ds, batch_size=1,
                             shuffle=False, num_workers=0)
    print(f"[test] Test samples: {n_test}")

    # ── Load CNN model
    ckpt  = torch.load(MODEL_PATH, map_location=DEVICE)
    model = AnimalClassifier(CLIP_DIM, num_classes).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[test] CNN model loaded (epoch {ckpt['epoch']})")

    # ── Evaluation counters
    cnn_correct      = 0
    clip_correct     = 0
    combined_correct = 0
    total            = 0

    class_cnn      = {i: [0, 0] for i in range(num_classes)}  # [correct, total]
    class_combined = {i: [0, 0] for i in range(num_classes)}

    print("\n[test] Evaluating — CNN  vs  CLIP zero-shot  vs  Combined …\n")

    with torch.no_grad():
        for feats, labels, _ in tqdm(test_loader, desc="Testing"):
            feat_cpu = feats.squeeze(0)          # (512,) on CPU
            feat_dev = feats.to(DEVICE)
            label    = labels.item()

            true_class = idx_to_class[label]

            # ── 1. CNN prediction
            logits     = model(feat_dev)
            probs      = softmax_probs(logits).squeeze(0).cpu()
            topk_p, topk_i = torch.topk(probs, k=5)
            cnn_preds  = [(idx_to_class[i.item()], p.item())
                          for p, i in zip(topk_p, topk_i)]
            cnn_top1   = cnn_preds[0][0]

            # ── 2. CLIP zero-shot prediction (from 25,000+ names)
            clip_results = zero_shot_predict(
                feat_cpu, text_matrix, animal_names, top_k=TOP_K_CLIP
            )
            clip_top1 = clip_results[0][0].lower().replace(" ", "_") \
                        if clip_results else ""

            # ── 3. Combined prediction (CNN + animals_data.txt via CLIP)
            combined = combined_predict(
                img_embedding  = feat_cpu,
                cnn_predictions= cnn_preds,
                text_matrix    = text_matrix,
                animal_names   = animal_names,
                idx_to_class   = idx_to_class,
                top_k          = 5,
            )
            combined_top1 = combined[0][0].replace(" ", "_") \
                            if combined else ""

            # ── Accuracy tracking
            true_norm = true_class.lower().replace("_", " ")

            if cnn_top1.lower().replace("_", " ") == true_norm:
                cnn_correct += 1
                class_cnn[label][0] += 1

            if clip_top1.lower().replace("_", " ") == true_norm:
                clip_correct += 1

            if combined_top1.lower().replace("_", " ") == true_norm:
                combined_correct += 1
                class_combined[label][0] += 1

            class_cnn[label][1]      += 1
            class_combined[label][1] += 1
            total += 1

    # ──────────────────────────────────────────────────────────
    # RESULTS
    # ──────────────────────────────────────────────────────────
    cnn_acc      = cnn_correct      / total
    clip_acc     = clip_correct     / total
    combined_acc = combined_correct / total

    print(f"\n{'='*65}")
    print(f"  TEST RESULTS  ({total} samples)")
    print(f"{'='*65}")
    print(f"  CNN alone          : {cnn_acc*100:>6.2f}%   ({cnn_correct}/{total})")
    print(f"  CLIP zero-shot     : {clip_acc*100:>6.2f}%   ({clip_correct}/{total})")
    print(f"  Combined (Option 2): {combined_acc*100:>6.2f}%   ({combined_correct}/{total})  ← BEST")
    print(f"{'='*65}")

    # Per-class report
    print(f"\n{'Class':<30} {'CNN':>8} {'Combined':>10}")
    print("-" * 52)
    for i in range(num_classes):
        cls  = idx_to_class[i]
        cc, ct = class_cnn[i]
        mc, mt = class_combined[i]
        c_acc  = cc / ct * 100 if ct else 0
        m_acc  = mc / mt * 100 if mt else 0
        flag   = " ✅" if m_acc > c_acc else ("" if m_acc == c_acc else " ⬇")
        print(f"  {cls:<28} {c_acc:>6.1f}%  {m_acc:>8.1f}%{flag}")

    print(f"\n[test] ✅ Done.")
    print(f"  CNN alone used       : 100-class image dataset")
    print(f"  CLIP zero-shot used  : animals_data.txt ({len(animal_names):,} names)")
    print(f"  Combined used        : BOTH data sources → Option 2")


if __name__ == "__main__":
    main()