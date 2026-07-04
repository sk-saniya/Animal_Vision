
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
    encode_image,
    get_clip,
    load_animals_txt,
)

# ──────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────
BASE_DIR       = Path(__file__).parent
DATASET_DIR    = BASE_DIR / "./Animal"
ANIMALS_TXT    = BASE_DIR / "animals_data.txt"
MODEL_OUT      = BASE_DIR / "cnn_model.pth"
LABEL_MAP_OUT  = BASE_DIR / "label_map.json"
EMBED_CACHE    = BASE_DIR / "text_embeddings_cache.pt"

BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 1e-3
VAL_SPLIT   = 0.15
SEED        = 42
CLIP_DIM    = 512

print(f"[train] Device: {DEVICE}")


# ──────────────────────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────────────────────
class AnimalImageDataset(Dataset):
    """
    Loads images from class-named sub-folders.
    Returns pre-computed CLIP image embeddings + integer labels.
    """
    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: Path, known_names: set):
        _, preprocess = get_clip()
        self.preprocess = preprocess

        # discover classes from folder names
        classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.idx_to_class = {i: c for c, i in self.class_to_idx.items()}
        self.num_classes  = len(classes)

        # log coverage against animals_data.txt
        matched = [c for c in classes
                   if c.lower().replace("_", " ") in known_names
                   or c.lower() in known_names]
        print(f"[train] Dataset classes          : {self.num_classes}")
        print(f"[train] Classes found in txt file: {len(matched)} / {self.num_classes}")

        # gather (path, label) pairs
        self.samples = []
        for cls in classes:
            for f in (root / cls).iterdir():
                if f.suffix.lower() in self.EXTS:
                    self.samples.append((f, self.class_to_idx[cls]))

        print(f"[train] Total images             : {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img  = Image.open(path).convert("RGB")
        feat = encode_image(img)          # (512,) CPU tensor
        return feat, label


# ──────────────────────────────────────────────────────────────
# MODEL — MLP head (CNN-style classifier on CLIP features)
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
# MAIN
# ──────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    # ── Step 1: Load animals_data.txt
    #    Build & cache text embeddings (used later by predict.py)
    print("\n[train] Step 1 — Processing animals_data.txt …")
    animal_names = load_animals_txt(ANIMALS_TXT)
    known_names  = {n.lower() for n in animal_names}

    print("[train] Pre-building CLIP text embedding cache for predict.py …")
    build_text_embeddings(animal_names, cache_path=EMBED_CACHE)
    print(f"[train] ✅ Text embedding cache ready → {EMBED_CACHE}")

    # ── Step 2: Image dataset
    print("\n[train] Step 2 — Loading image dataset …")
    dataset = AnimalImageDataset(DATASET_DIR, known_names)

    n_val   = int(len(dataset) * VAL_SPLIT)
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, num_workers=0)

    print(f"[train] Train: {n_train}  |  Val: {n_val}")

    # ── Save label map
    with open(LABEL_MAP_OUT, "w") as f:
        json.dump(dataset.idx_to_class, f, indent=2)
    print(f"[train] Label map → {LABEL_MAP_OUT}")

    # ── Step 3: Build model
    model     = AnimalClassifier(CLIP_DIM, dataset.num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    print(f"\n[train] Step 3 — Training MLP head for {EPOCHS} epochs …")
    print(f"         Architecture: CLIP(frozen) → MLP(512→256→{dataset.num_classes})")
    print(f"         Both data sources active:")
    print(f"           • animals/ images    → CrossEntropy training loss")
    print(f"           • animals_data.txt   → text embedding cache for predict.py\n")

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── train
        model.train()
        run_loss = correct = total = 0
        for feats, labels in tqdm(train_loader,
                                  desc=f"Epoch {epoch:>2}/{EPOCHS} [Train]",
                                  leave=False):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            logits = model(feats)
            loss   = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            run_loss += loss.item() * feats.size(0)
            correct  += (logits.argmax(1) == labels).sum().item()
            total    += feats.size(0)

        train_loss = run_loss / total
        train_acc  = correct  / total

        # ── validate
        model.eval()
        v_loss = v_correct = v_total = 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                logits = model(feats)
                v_loss    += criterion(logits, labels).item() * feats.size(0)
                v_correct += (logits.argmax(1) == labels).sum().item()
                v_total   += feats.size(0)

        val_loss = v_loss    / v_total
        val_acc  = v_correct / v_total
        scheduler.step()

        print(f"Epoch {epoch:>3}/{EPOCHS}  "
              f"Train Loss {train_loss:.4f}  Acc {train_acc:.4f}  |  "
              f"Val Loss {val_loss:.4f}  Acc {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "num_classes": dataset.num_classes,
                "clip_dim":    CLIP_DIM,
            }, MODEL_OUT)
            print(f"  ✅ Best model saved  (val_acc = {best_val_acc:.4f})")

    print(f"\n[train] ✅ Done!  Best val accuracy : {best_val_acc*100:.2f}%")
    print(f"         Model weights  → {MODEL_OUT}")
    print(f"         Label map      → {LABEL_MAP_OUT}")
    print(f"         Text emb cache → {EMBED_CACHE}")


if __name__ == "__main__":
    main()
