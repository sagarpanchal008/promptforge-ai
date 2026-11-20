"""
==============================================================================
FILE: models/train_cnn.py
PURPOSE:
    Trains and evaluates the 1D-CNN model using preprocessed data splits.
    Can be run in Google Colab (GPU accelerated) or locally with CPU/GPU.
    Computes genuine training/validation metrics, confusion matrix, precision,
    recall, and F1-score on the independent test set.

HOW IT CONNECTS TO THE PROJECT:
    - Ingests 'dataset/preprocessed_splits.json' generated in Phase 2.
    - Saves the trained model to 'models/cnn_model.keras'.
    - Saves evaluation statistics to 'models/cnn_metrics.json'.
    - Serves as the deep learning evaluation engine for Agent 2 in Phase 6.

EXECUTION:
    python models/train_cnn.py
==============================================================================
"""

import json
import os
import sys

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SPLITS_JSON = os.path.join(PROJECT_ROOT, "dataset", "preprocessed_splits.json")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_SAVE_PATH = os.path.join(MODELS_DIR, "cnn_model.keras")
METRICS_SAVE_PATH = os.path.join(MODELS_DIR, "cnn_metrics.json")
PLOT_SAVE_PATH = os.path.join(MODELS_DIR, "cnn_training_curves.png")

# Ensure models/ directory is in sys.path
if MODELS_DIR not in sys.path:
    sys.path.append(MODELS_DIR)


def train_cnn():
    print("=" * 75)
    print("PHASE 3: 1D-CONVOLUTIONAL NEURAL NETWORK (CNN) TRAINING PIPELINE")
    print("=" * 75)

    # 1. Dependency Check
    try:
        import numpy as np
        import tensorflow as tf
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
        from cnn_model import build_cnn_model
    except ImportError as e:
        print(f"[!] Missing required dependency: {e}")
        print("[*] To train locally, install: pip install tensorflow numpy matplotlib scikit-learn")
        print("[*] Or run 'notebooks/02_CNN_training.ipynb' directly in Google Colab (free GPU).")
        return False

    # 2. Check Data
    if not os.path.exists(SPLITS_JSON):
        print(f"[ERROR] Preprocessed splits not found at: {SPLITS_JSON}")
        print("[*] Run 'python utils/preprocessing.py' first.")
        return False

    # 3. Load preprocessed tensors
    print(f"[*] Loading preprocessed splits from: {SPLITS_JSON}")
    with open(SPLITS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = data["metadata"]
    vocab_size = meta["vocab_size"]
    max_len = meta["max_len"]
    num_classes = meta["num_classes"]
    classes = meta["classes"]

    X_train = np.array(data["train"]["X"], dtype=np.int32)
    y_train = np.array(data["train"]["y_one_hot"], dtype=np.float32)
    y_train_idx = np.array(data["train"]["y"], dtype=np.int32)

    X_val = np.array(data["val"]["X"], dtype=np.int32)
    y_val = np.array(data["val"]["y_one_hot"], dtype=np.float32)
    y_val_idx = np.array(data["val"]["y"], dtype=np.int32)

    X_test = np.array(data["test"]["X"], dtype=np.int32)
    y_test = np.array(data["test"]["y_one_hot"], dtype=np.float32)
    y_test_idx = np.array(data["test"]["y"], dtype=np.int32)

    print(f"[+] Dataset Split Summary:")
    print(f"    - Training Shape   : X={X_train.shape}, y={y_train.shape}")
    print(f"    - Validation Shape : X={X_val.shape}, y={y_val.shape}")
    print(f"    - Test Shape       : X={X_test.shape}, y={y_test.shape}")
    print(f"    - Vocabulary Size  : {vocab_size} unique tokens")
    print(f"    - Classes          : {classes} (Count: {num_classes})")

    # 4. Build Model
    print("\n[*] Constructing 1D-CNN Architecture...")
    model = build_cnn_model(
        vocab_size=vocab_size,
        embedding_dim=128,
        max_len=max_len,
        num_classes=num_classes,
        dropout_rate=0.4,
    )
    model.summary()

    # 5. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-5,
            verbose=1,
        ),
    ]

    # 6. Model Training
    print("\n[*] Starting 1D-CNN Training (Max 25 Epochs, Batch Size: 32)...")
    epochs = 25
    batch_size = 32

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )

    # 7. Independent Test Set Evaluation
    print("\n" + "=" * 75)
    print("EVALUATION ON INDEPENDENT TEST SET (189 SAMPLES)")
    print("=" * 75)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[+] Test Loss     : {test_loss:.4f}")
    print(f"[+] Test Accuracy : {test_acc * 100:.2f}%")

    # Generate predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Calculate Confusion Matrix
    from collections import defaultdict
    cm = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true_lbl, pred_lbl in zip(y_test_idx, y_pred):
        cm[true_lbl][pred_lbl] += 1

    print("\n[+] Confusion Matrix (Rows: Actual, Columns: Predicted):")
    print(f"    {'':<10} | {'Pred Poor':<12} | {'Pred Avg':<12} | {'Pred Good':<12}")
    print("    " + "-" * 50)
    for i, c_name in enumerate(classes):
        print(f"    {c_name:<10} | {cm[i][0]:<12} | {cm[i][1]:<12} | {cm[i][2]:<12}")

    # Calculate Precision, Recall, F1-Score per class
    metrics_report = {}
    print("\n[+] Classification Report:")
    print(f"    {'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Support':<8}")
    print("    " + "-" * 56)

    total_f1 = 0
    total_prec = 0
    total_rec = 0

    for i, c_name in enumerate(classes):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(num_classes) if r != i)
        fn = sum(cm[i][c] for c in range(num_classes) if c != i)
        support = sum(cm[i])

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        total_prec += prec
        total_rec += rec
        total_f1 += f1

        metrics_report[c_name] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "support": support,
        }
        print(f"    {c_name:<12} | {prec * 100:>8.2f}% | {rec * 100:>8.2f}% | {f1 * 100:>8.2f}% | {support:>8}")

    macro_f1 = total_f1 / num_classes
    macro_prec = total_prec / num_classes
    macro_rec = total_rec / num_classes
    print("    " + "-" * 56)
    print(f"    {'Macro Avg':<12} | {macro_prec * 100:>8.2f}% | {macro_rec * 100:>8.2f}% | {macro_f1 * 100:>8.2f}% | {len(y_test_idx):>8}")

    # 8. Save Metrics JSON
    results_summary = {
        "model": "1D-CNN",
        "test_loss": round(float(test_loss), 4),
        "test_accuracy": round(float(test_acc), 4),
        "macro_precision": round(float(macro_prec), 4),
        "macro_recall": round(float(macro_rec), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": cm,
        "class_metrics": metrics_report,
        "training_history": {
            "loss": [round(float(v), 4) for v in history.history["loss"]],
            "val_loss": [round(float(v), 4) for v in history.history["val_loss"]],
            "accuracy": [round(float(v), 4) for v in history.history["accuracy"]],
            "val_accuracy": [round(float(v), 4) for v in history.history["val_accuracy"]],
        },
    }

    with open(METRICS_SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=4)
    print(f"\n[+] Saved CNN Evaluation Metrics to: {METRICS_SAVE_PATH}")

    # 9. Plot curves if matplotlib is available
    try:
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history.history["loss"]) + 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        ax1.plot(epochs_range, history.history["loss"], "b-o", label="Training Loss")
        ax1.plot(epochs_range, history.history["val_loss"], "r-o", label="Validation Loss")
        ax1.set_title("1D-CNN Loss vs Epochs", fontsize=14)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Categorical Crossentropy")
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend()

        ax2.plot(epochs_range, history.history["accuracy"], "b-o", label="Training Accuracy")
        ax2.plot(epochs_range, history.history["val_accuracy"], "r-o", label="Validation Accuracy")
        ax2.set_title("1D-CNN Accuracy vs Epochs", fontsize=14)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(PLOT_SAVE_PATH, dpi=300)
        print(f"[+] Saved CNN Loss & Accuracy Curves to: {PLOT_SAVE_PATH}")
    except Exception as e:
        print(f"[*] Note: Matplotlib plot skipped ({e}). Metrics saved in JSON.")

    print("\n" + "=" * 75)
    print(f"[SUCCESS] 1D-CNN Model trained and saved to: {MODEL_SAVE_PATH}")
    print("=" * 75)
    return True


if __name__ == "__main__":
    train_cnn()
