"""
Evaluate a trained LungCNET checkpoint on the held-out test set.

Reproduces the metrics reported in the paper's Tables 3-6: per-class
precision/recall/F1 (benign, malignant, normal) and the macro-averaged
F1-score, plus a confusion matrix plot.

Usage:
    python -m src.evaluate --model checkpoints/lungcnet_best.h5 --config configs/config.yaml
"""

import argparse
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import yaml
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from tensorflow.keras.models import load_model

from src.data import load_and_split_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LungCNET.")
    parser.add_argument("--model", type=str, required=True, help="Path to a saved .h5 model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluate"]
    class_names = model_cfg["class_names"]

    print("Loading test set...")
    (_, _), (_, _), (X_test, y_test), _ = load_and_split_dataset(cfg)

    print(f"Loading model from {args.model} ...")
    model = load_model(args.model)

    print("Running inference on test set...")
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    print("\n=== Classification Report (per-class precision/recall/F1) ===")
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(report)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Macro-averaged F1-score: {macro_f1 * 100:.2f}%")

    output_dir = pathlib.Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "classification_report.txt", "w") as f:
        f.write(report)
        f.write(f"\nMacro-averaged F1-score: {macro_f1 * 100:.2f}%\n")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix - LungCNET")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=150)
    print(f"\nSaved classification report and confusion matrix to {output_dir}/")


if __name__ == "__main__":
    main()
