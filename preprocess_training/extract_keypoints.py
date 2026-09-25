"""
Input  : Data/raw/classroom_activity/videos/<kelas>/*.mp4
Output : Data/processed_npy/<kelas>/<nama_video>.npy

Setiap file .npy berisi array shape (N_frames, 34)
  - 17 keypoints MoveNet × 2 (x, y) yang sudah dinormalisasi ke hip
  - Derived features: joint angles + wrist velocity

Jalankan sekali sebelum training:
  python preprocess_training/extract_keypoints.py
"""

import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from pathlib import Path

# ── Config ────────────────────────────────────────────
BASE_DIR     = Path(__file__).resolve().parents[1]
RAW_DIR      = BASE_DIR / "Data" / "raw" / "classroom_activity" / "videos"
OUTPUT_DIR   = BASE_DIR / "Data" / "processed_npy"
MODEL_URL    = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

CLASSES = [
    "hand_raise",
    "holding_book",
    "holding_mobile_phone",
    "reading_book",
    "writing",
    "memperhatikan",   # tambah/sesuaikan dengan folder
]

# MoveNet keypoint indices (17 joints)
KP = {
    "nose": 0, "left_eye": 1, "right_eye": 2,
    "left_ear": 3, "right_ear": 4,
    "left_shoulder": 5, "right_shoulder": 6,
    "left_elbow": 7, "right_elbow": 8,
    "left_wrist": 9, "right_wrist": 10,
    "left_hip": 11, "right_hip": 12,
    "left_knee": 13, "right_knee": 14,
    "left_ankle": 15, "right_ankle": 16,
}

# ── Load MoveNet ──────────────────────────────────────
print("Loading MoveNet...")
module   = hub.load(MODEL_URL)
movenet  = module.signatures["serving_default"]
print("MoveNet loaded.")


# ── Helpers ───────────────────────────────────────────
def run_movenet(frame_rgb: np.ndarray) -> np.ndarray:
    """Jalankan MoveNet, return keypoints shape (17, 3) → [y, x, confidence]."""
    img    = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, 0), 192, 192)
    img    = tf.cast(img, tf.int32)
    output = movenet(img)
    kps    = output["output_0"].numpy()[0, 0]   # (17, 3)
    return kps


def normalize_to_hip(kps: np.ndarray) -> np.ndarray:
    """
    Normalisasi koordinat relatif ke midpoint hip.
    Menghilangkan efek posisi absolut di frame.
    kps shape: (17, 3) → return (17, 2) koordinat ternormalisasi.
    """
    hip_mid = (kps[KP["left_hip"], :2] + kps[KP["right_hip"], :2]) / 2.0
    coords  = kps[:, :2] - hip_mid   # (17, 2)

    # Scale normalization: jarak bahu kiri-kanan sebagai referensi skala
    shoulder_dist = np.linalg.norm(
        kps[KP["left_shoulder"], :2] - kps[KP["right_shoulder"], :2]
    )
    if shoulder_dist > 1e-6:
        coords /= shoulder_dist

    return coords   # (17, 2)


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Sudut di titik b antara vektor ba dan bc (derajat)."""
    ba = a - b
    bc = c - b
    n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos = np.dot(ba, bc) / (n1 * n2)
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def extract_features(kps_norm: np.ndarray, kps_prev: np.ndarray | None) -> np.ndarray:
    """
    Gabungkan: normalized keypoints (34) + angles (6) + wrist velocity (2)
    Total: 42 features per frame
    """
    flat = kps_norm.flatten()   # (34,)

    k = kps_norm
    angles = np.array([
        joint_angle(k[KP["left_shoulder"]],  k[KP["left_elbow"]],  k[KP["left_wrist"]]),
        joint_angle(k[KP["right_shoulder"]], k[KP["right_elbow"]], k[KP["right_wrist"]]),
        joint_angle(k[KP["left_hip"]],       k[KP["left_shoulder"]], k[KP["left_elbow"]]),
        joint_angle(k[KP["right_hip"]],      k[KP["right_shoulder"]], k[KP["right_elbow"]]),
        joint_angle(k[KP["left_elbow"]],     k[KP["left_shoulder"]], k[KP["left_hip"]]),
        joint_angle(k[KP["right_elbow"]],    k[KP["right_shoulder"]], k[KP["right_hip"]]),
    ], dtype=np.float32)   # (6,)

    if kps_prev is not None:
        vel_l = np.linalg.norm(k[KP["left_wrist"]]  - kps_prev[KP["left_wrist"]])
        vel_r = np.linalg.norm(k[KP["right_wrist"]] - kps_prev[KP["right_wrist"]])
    else:
        vel_l, vel_r = 0.0, 0.0

    velocity = np.array([vel_l, vel_r], dtype=np.float32)   # (2,)

    return np.concatenate([flat, angles, velocity])   # (42,)


def is_pose_valid(kps_raw: np.ndarray, min_confidence: float = 0.3) -> bool:
    """Cek apakah keypoints kritis memiliki confidence cukup."""
    critical = [
        KP["left_shoulder"], KP["right_shoulder"],
        KP["left_elbow"],    KP["right_elbow"],
        KP["left_wrist"],    KP["right_wrist"],
    ]
    return all(kps_raw[i, 2] >= min_confidence for i in critical)


def extract_video(video_path: Path) -> np.ndarray | None:
    """
    Ekstrak semua frame valid dari satu video.
    Return: (N_valid_frames, 42) atau None jika video tidak bisa dibaca.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [SKIP] Tidak bisa buka: {video_path.name}")
        return None

    features_list = []
    kps_prev_norm = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        kps_raw   = run_movenet(frame_rgb)          # (17, 3)

        if not is_pose_valid(kps_raw):
            kps_prev_norm = None                    # reset velocity jika frame invalid
            continue

        kps_norm = normalize_to_hip(kps_raw)        # (17, 2)
        feat     = extract_features(kps_norm, kps_prev_norm)   # (42,)
        features_list.append(feat)
        kps_prev_norm = kps_norm

    cap.release()

    if len(features_list) < 10:
        print(f"  [SKIP] Frame valid terlalu sedikit ({len(features_list)}): {video_path.name}")
        return None

    return np.array(features_list, dtype=np.float32)   # (N, 42)


# ── Main ──────────────────────────────────────────────
def main():
    total_ok   = 0
    total_skip = 0

    for cls in CLASSES:
        video_dir  = RAW_DIR / cls
        output_cls = OUTPUT_DIR / cls
        output_cls.mkdir(parents=True, exist_ok=True)

        if not video_dir.exists():
            print(f"[WARN] Folder tidak ditemukan: {video_dir}")
            continue

        videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
        print(f"\n[{cls}] {len(videos)} video ditemukan")

        for vp in videos:
            out_path = output_cls / (vp.stem + ".npy")
            if out_path.exists():
                print(f"  [EXISTS] {vp.name} — skip")
                continue

            print(f"  Processing: {vp.name} ...", end=" ", flush=True)
            feat = extract_video(vp)

            if feat is not None:
                np.save(out_path, feat)
                print(f"OK ({feat.shape[0]} frames)")
                total_ok += 1
            else:
                total_skip += 1

    print(f"\n=== Selesai: {total_ok} berhasil, {total_skip} dilewati ===")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
