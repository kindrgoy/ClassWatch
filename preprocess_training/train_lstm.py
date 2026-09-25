"""
Input  : Data/processed_npy/dataset.npz  (output Step 2)
Output : Data/models/lstm_activity.keras
         Data/models/label_map.json
         Data/logs/training_<timestamp>.json

Jalankan:
  python preprocess_training/train_lstm.py
"""

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend (server/Jetson)
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

import tensorflow as tf
import keras
from keras import layers, callbacks, optimizers
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

# ── Config ────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]
DATASET     = BASE_DIR / "Data" / "processed_npy" / "dataset.npz"
MODEL_DIR   = BASE_DIR / "Data" / "models"
LOG_DIR     = BASE_DIR / "Data" / "logs"
MODEL_PATH  = MODEL_DIR / "lstm_activity.keras"
LABEL_PATH  = MODEL_DIR / "label_map.json"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Hyperparameter — tuning di sini
LSTM_UNITS   = 128       # cukup untuk edge device
DROPOUT      = 0.4
DENSE_UNITS  = 64
BATCH_SIZE   = 32
EPOCHS       = 80
LR           = 1e-3
PATIENCE     = 12        # early stopping


# ── Load data ─────────────────────────────────────────
print("Loading dataset...")
data        = np.load(DATASET, allow_pickle=True)
X_train     = data["X_train"]          # (N, 30, 42)
y_train     = data["y_train"]
X_val       = data["X_val"]
y_val       = data["y_val"]
X_test      = data["X_test"]
y_test      = data["y_test"]
class_names = list(data["class_names"])
N_CLASSES   = len(class_names)

print(f"Kelas          : {class_names}")
print(f"Train/Val/Test : {len(X_train)} / {len(X_val)} / {len(X_test)}")
print(f"Input shape    : {X_train.shape[1:]}")


# ── Class weights (tangani imbalanced dataset) ────────
from sklearn.utils.class_weight import compute_class_weight
cw_values = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
class_weight = dict(enumerate(cw_values))
print(f"Class weights  : { {class_names[k]: round(v, 2) for k, v in class_weight.items()} }")


# ── Model ─────────────────────────────────────────────
def build_model(seq_len: int, n_features: int, n_classes: int) -> keras.Model:
    """
    Arsitektur: LSTM → Dropout → Dense → Output
    Ringkas dan kompatibel dengan Jetson (TensorRT-friendly).
    """
    inputs = keras.Input(shape=(seq_len, n_features), name="pose_sequence")

    # Layer LSTM pertama — return sequence untuk stacking
    x = layers.LSTM(LSTM_UNITS, return_sequences=True, name="lstm_1")(inputs)
    x = layers.Dropout(DROPOUT, name="drop_1")(x)

    # Layer LSTM kedua — ambil output terakhir
    x = layers.LSTM(LSTM_UNITS // 2, return_sequences=False, name="lstm_2")(x)
    x = layers.Dropout(DROPOUT, name="drop_2")(x)

    x = layers.Dense(DENSE_UNITS, activation="relu", name="dense_1")(x)
    x = layers.Dropout(DROPOUT / 2, name="drop_3")(x)

    outputs = layers.Dense(n_classes, activation="softmax", name="output")(x)

    model = keras.Model(inputs, outputs, name="ClassWatch_LSTM")
    model.compile(
        optimizer=optimizers.Adam(learning_rate=LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


_, SEQ_LEN, N_FEATURES = X_train.shape
model = build_model(SEQ_LEN, N_FEATURES, N_CLASSES)
model.summary()


# ── Callbacks ─────────────────────────────────────────
cb_list = [
    callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE,
        restore_best_weights=True, verbose=1
    ),
    callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=PATIENCE // 2, min_lr=1e-6, verbose=1
    ),
    callbacks.ModelCheckpoint(
        str(MODEL_PATH), monitor="val_loss",
        save_best_only=True, verbose=1
    ),
]


# ── Training ──────────────────────────────────────────
print("\n=== Training dimulai ===")
t0 = time.time()

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    class_weight=class_weight,
    callbacks=cb_list,
    verbose=1,
)

duration = round(time.time() - t0)
print(f"\n=== Training selesai ({duration}s) ===")


# ── Evaluasi ──────────────────────────────────────────
print("\n--- Evaluasi Test Set ---")
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = np.argmax(y_pred_prob, axis=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
f1_weighted = f1_score(y_test, y_pred, average="weighted")
cm          = confusion_matrix(y_test, y_pred)

print(f"Test Accuracy  : {test_acc:.4f}")
print(f"F1 Weighted    : {f1_weighted:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=class_names))
print("Confusion Matrix:")
print(cm)


# ── Simpan label map ──────────────────────────────────
label_map = {str(i): name for i, name in enumerate(class_names)}
with open(LABEL_PATH, "w") as f:
    json.dump(label_map, f, indent=2)
print(f"\nLabel map disimpan: {LABEL_PATH}")


# ── Simpan log training ───────────────────────────────
log_payload = {
    "timestamp":       datetime.now().isoformat(),
    "duration_seconds": duration,
    "epochs_run":      len(history.history["loss"]),
    "test_accuracy":   round(float(test_acc), 4),
    "f1_weighted":     round(float(f1_weighted), 4),
    "test_loss":       round(float(test_loss), 4),
    "hyperparameters": {
        "lstm_units":  LSTM_UNITS,
        "dropout":     DROPOUT,
        "dense_units": DENSE_UNITS,
        "batch_size":  BATCH_SIZE,
        "lr":          LR,
        "seq_len":     SEQ_LEN,
        "n_features":  N_FEATURES,
    },
    "class_names": class_names,
    "confusion_matrix": cm.tolist(),
}
log_file = LOG_DIR / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(log_file, "w") as f:
    json.dump(log_payload, f, indent=2)
print(f"Log disimpan   : {log_file}")


# ── Plot training history ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Training History", fontsize=13)

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history["accuracy"],     label="Train Acc")
axes[1].plot(history.history["val_accuracy"], label="Val Acc")
axes[1].set_title("Accuracy")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plot_path = LOG_DIR / f"training_plot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
plt.tight_layout()
plt.savefig(plot_path, dpi=120)
print(f"Plot disimpan  : {plot_path}")

print(f"\nModel disimpan : {MODEL_PATH}")
