import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# paths
PROJECT_ROOT = Path(r"C:\Users\maksi\Desktop\fer\3. godina apsolvent\zavrsni rad\project")
MODELS_DIR = PROJECT_ROOT / "models"
OUT_DIR = PROJECT_ROOT / "reports" / "metrics"


# helpers
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def sanitize(s: str) -> str:
    return (
        s.replace(" ", "_")
         .replace("/", "_")
         .replace("\\", "_")
         .replace(":", "_")
    )


def plot_confusion_matrix(cm, title, out_path):
    plt.figure(figsize=(4.8, 4.2))
    im = plt.imshow(cm, cmap="Blues")
    plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])
    plt.title(title)

    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            plt.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11
            )

    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", choices=["base", "enhanced"], required=True)
    parser.add_argument("--min_n", type=int, default=30, help="min samples per instrument")
    args = parser.parse_args()

    run_dir = MODELS_DIR / f"ml_output_global_{args.feature_set}"
    cls_path = run_dir / "classification_predictions.csv"

    if not cls_path.exists():
        raise FileNotFoundError(cls_path)

    df = pd.read_csv(cls_path)

    out_root = OUT_DIR / args.feature_set / "confusion_matrices_per_instrument"
    ensure_dir(out_root)

    print("======================================")
    print("CONFUSION MATRICES PER INSTRUMENT")
    print("Feature set:", args.feature_set)
    print("Input:", cls_path)
    print("Output:", out_root)
    print("======================================")

    for target, df_t in df.groupby("target"):
        target_dir = out_root / sanitize(target)
        ensure_dir(target_dir)

        for inst, g in df_t.groupby("Instrument"):
            if len(g) < args.min_n:
                continue

            y_true = g["y_true"].astype(int).to_numpy()
            y_pred = g["y_pred"].astype(int).to_numpy()

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

            title = f"{inst} – {target}\n(n={len(g)})"
            fname = f"cm_{sanitize(inst)}_{sanitize(target)}.png"
            out_path = target_dir / fname

            plot_confusion_matrix(cm, title, out_path)

    print("\n✓ DONE: Confusion matrices saved.")
    return 0


if __name__ == "__main__":
    main()
