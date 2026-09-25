"""
Input  : Data/processed_npy/<kelas>/*.npy  (output Step 1)
Output : Data/processed_npy/dataset.npz

dataset.npz berisi:
  X_train, X_val, X_test  → shape (N, 30, 42)
  y_train, y_val, y_test  → shape (N,) int label
  class_names             → list nama kelas

Jalankan setelah extract_keypoints.py:
  python preprocess_training/build_dataset.py
"""

import os
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter

# ── Config ────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
NPY_DIR      = BASE_DIR / "Data" / "processed_npy"
OUTPUT_FILE  = NPY_DIR / "dataset.npz"

CLASSES = [
    "hand_raise",
    "holding_book",
    "holding_mobile_phone",
    "reading_book",
    "writing",
    "memperhatikan",
]

SEQ_LEN  = 30    # panjang window (frame) — sesuai flowchart kamu
STRIDE   = 10    # sliding window step; lebih kecil = lebih banyak sampel
                 # stride=1 untuk dataset kecil, stride=10 untuk >50 video/kelas

VAL_SIZE  = 0.15
TEST_SIZE = 0.15
SEED      = 42


# ── Sliding window ────────────────────────────────────
def make_sequences(features: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """
    Potong array (N, F) menjadi (M, seq_len, F) dengan sliding window.
    M = jumlah window yang bisa dibuat.
    """
    seqs = []
    for start in range(0, len(features) - seq_len + 1, stride):
        seqs.append(features[start : start + seq_len])
    return np.array(seqs, dtype=np.float32) if seqs else np.empty((0, seq_len, features.shape[1]))


# ── Main ──────────────────────────────────────────────
def main():
    X_all, y_all = [], []
    class_names  = []

    for label_idx, cls in enumerate(CLASSES):
        cls_dir = NPY_DIR / cls
        if not cls_dir.exists():
            print(f"[WARN] Folder tidak ditemukan: {cls_dir} — dilewati")
            continue

        npy_files = list(cls_dir.glob("*.npy"))
        if not npy_files:
            print(f"[WARN] Tidak ada .npy di {cls_dir} — jalankan extract_keypoints.py dulu")
            continue

        class_names.append(cls)
        cls_seqs = 0

        for npy_path in npy_files:
            features = np.load(npy_path)   # (N, 42)
            seqs     = make_sequences(features, SEQ_LEN, STRIDE)

            if len(seqs) == 0:
                continue

            X_all.append(seqs)
            y_all.extend([label_idx] * len(seqs))
            cls_seqs += len(seqs)

        print(f"[{cls}] {len(npy_files)} video → {cls_seqs} sequences")

    if not X_all:
        print("ERROR: Tidak ada data. Pastikan extract_keypoints.py sudah dijalankan.")
        return

    X = np.concatenate(X_all, axis=0)   # (Total, 30, 42)
    y = np.array(y_all, dtype=np.int32) # (Total,)

    print(f"\nTotal sequences : {len(X)}")
    print(f"Shape X         : {X.shape}")
    print(f"Distribusi kelas: {dict(Counter(y))}")

    # ── Stratified split ─────────────────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=VAL_SIZE + TEST_SIZE,
        stratify=y, random_state=SEED
    )
    val_ratio = VAL_SIZE / (VAL_SIZE + TEST_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=1 - val_ratio,
        stratify=y_temp, random_state=SEED
    )

    print(f"\nSplit:")
    print(f"  Train : {len(X_train)}")
    print(f"  Val   : {len(X_val)}")
    print(f"  Test  : {len(X_test)}")

    # ── Simpan ───────────────────────────────────────
    np.savez_compressed(
        OUTPUT_FILE,
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        class_names=np.array(class_names),
    )
    print(f"\nDataset disimpan: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
