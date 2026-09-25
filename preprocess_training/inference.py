"""
Digunakan oleh routes.py untuk klasifikasi real-time.

Cara integrasi ke routes.py:
    from model.inference import ActivityClassifier
    classifier = ActivityClassifier()           # load sekali saat startup

    # Di dalam classify_frame():
    result = classifier.predict(frame)
"""

import json
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import keras
from collections import deque
from pathlib import Path

# ── Config ────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]
MODEL_PATH  = BASE_DIR / "Data" / "models" / "lstm_activity.keras"
LABEL_PATH  = BASE_DIR / "Data" / "models" / "label_map.json"
MOVENET_URL = "https://tfhub.dev/google/movenet/singlepose/lightning/4"

SEQ_LEN           = 30
CONFIDENCE_THRESH = 0.6    # threshold dari flowchart kamu

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


class ActivityClassifier:
    """
    Wrapper inference untuk satu orang.
    Untuk multi-person: buat satu instance per orang yang terdeteksi.
    """

    def __init__(self):
        print("[Classifier] Loading MoveNet...")
        module       = hub.load(MOVENET_URL)
        self._movenet = module.signatures["serving_default"]

        print("[Classifier] Loading LSTM model...")
        self._lstm   = keras.models.load_model(str(MODEL_PATH))

        with open(LABEL_PATH) as f:
            lm = json.load(f)
        self._labels = {int(k): v for k, v in lm.items()}

        self._buffer      : deque         = deque(maxlen=SEQ_LEN)
        self._last_status : str           = "memperhatikan"
        self._last_conf   : float         = 0.0
        self._kps_prev    : np.ndarray | None = None

        print(f"[Classifier] Siap. Kelas: {list(self._labels.values())}")

    # ── MoveNet ───────────────────────────────────────
    def _run_movenet(self, frame_rgb: np.ndarray) -> np.ndarray:
        img    = tf.image.resize_with_pad(tf.expand_dims(frame_rgb, 0), 192, 192)
        img    = tf.cast(img, tf.int32)
        output = self._movenet(img)
        return output["output_0"].numpy()[0, 0]   # (17, 3)

    def _is_valid(self, kps: np.ndarray, min_conf: float = 0.3) -> bool:
        critical = [KP["left_shoulder"], KP["right_shoulder"],
                    KP["left_elbow"],    KP["right_elbow"],
                    KP["left_wrist"],    KP["right_wrist"]]
        return all(kps[i, 2] >= min_conf for i in critical)

    def _normalize(self, kps: np.ndarray) -> np.ndarray:
        hip_mid = (kps[KP["left_hip"], :2] + kps[KP["right_hip"], :2]) / 2.0
        coords  = kps[:, :2] - hip_mid
        dist    = np.linalg.norm(kps[KP["left_shoulder"], :2] - kps[KP["right_shoulder"], :2])
        if dist > 1e-6:
            coords /= dist
        return coords

    def _angles(self, k: np.ndarray) -> np.ndarray:
        def ang(a, b, c):
            ba, bc = a - b, c - b
            n1, n2 = np.linalg.norm(ba), np.linalg.norm(bc)
            if n1 < 1e-6 or n2 < 1e-6: return 0.0
            return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / (n1 * n2), -1, 1))))
        return np.array([
            ang(k[KP["left_shoulder"]],  k[KP["left_elbow"]],   k[KP["left_wrist"]]),
            ang(k[KP["right_shoulder"]], k[KP["right_elbow"]],  k[KP["right_wrist"]]),
            ang(k[KP["left_hip"]],       k[KP["left_shoulder"]], k[KP["left_elbow"]]),
            ang(k[KP["right_hip"]],      k[KP["right_shoulder"]], k[KP["right_elbow"]]),
            ang(k[KP["left_elbow"]],     k[KP["left_shoulder"]], k[KP["left_hip"]]),
            ang(k[KP["right_elbow"]],    k[KP["right_shoulder"]], k[KP["right_hip"]]),
        ], dtype=np.float32)

    def _extract_features(self, kps_norm: np.ndarray) -> np.ndarray:
        flat = kps_norm.flatten()
        angs = self._angles(kps_norm)
        if self._kps_prev is not None:
            vel = np.array([
                np.linalg.norm(kps_norm[KP["left_wrist"]]  - self._kps_prev[KP["left_wrist"]]),
                np.linalg.norm(kps_norm[KP["right_wrist"]] - self._kps_prev[KP["right_wrist"]]),
            ], dtype=np.float32)
        else:
            vel = np.zeros(2, dtype=np.float32)
        return np.concatenate([flat, angs, vel])   # (42,)

    # ── Predict ───────────────────────────────────────
    def predict(self, frame_bgr: np.ndarray) -> dict:
        """
        Input  : frame BGR (np.ndarray)
        Output : dict siap kirim ke WebSocket frontend
        """
        import cv2
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        kps_raw   = self._run_movenet(frame_rgb)

        if not self._is_valid(kps_raw):
            # Pose tidak valid — pertahankan status sebelumnya
            self._kps_prev = None
            return self._build_response(detected=0)

        kps_norm = self._normalize(kps_raw)
        feat     = self._extract_features(kps_norm)
        self._buffer.append(feat)
        self._kps_prev = kps_norm

        # Tunggu buffer penuh (30 frame)
        if len(self._buffer) < SEQ_LEN:
            return self._build_response(detected=1)

        # Inferensi LSTM
        seq        = np.array(self._buffer, dtype=np.float32)[np.newaxis]   # (1, 30, 42)
        probs      = self._lstm.predict(seq, verbose=0)[0]
        confidence = float(np.max(probs))
        pred_label = self._labels[int(np.argmax(probs))]

        if confidence >= CONFIDENCE_THRESH:
            self._last_status = pred_label
            self._last_conf   = confidence

        return self._build_response(detected=1)

    def _build_response(self, detected: int) -> dict:
        """Format output sesuai yang diharapkan frontend."""
        act_keys = list(self._labels.values())
        activities = {k: 0 for k in act_keys}

        if detected > 0:
            activities[self._last_status] = 100

        # Hitung engagement index
        learn_keys = {"memperhatikan", "reading_book", "holding_book", "writing", "hand_raise"}
        engagement = 100 if self._last_status in learn_keys else 0

        return {
            "detected":         detected,
            "engagement_index": engagement,
            "activities":       activities,
            "confidence":       round(self._last_conf, 3),
            "current_activity": self._last_status,
        }
