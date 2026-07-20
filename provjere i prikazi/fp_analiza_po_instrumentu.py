import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


# -----------------------------
# ARGUMENTI
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python fp_analiza_po_instrumentu.py <target> <model_type>")
    print("Example: python fp_analiza_po_instrumentu.py Down10 enhanced")
    sys.exit(1)

TARGET = sys.argv[1]
MODEL_TYPE = sys.argv[2]

if MODEL_TYPE not in ["base", "enhanced"]:
    print("model_type must be 'base' or 'enhanced'")
    sys.exit(1)


# -----------------------------
# PATHS
# -----------------------------
pred_path = f"models/ml_output_global_{MODEL_TYPE}/classification_predictions.csv"
analysis_dir = f"models/ml_output_global_{MODEL_TYPE}/analysis_{TARGET}"
os.makedirs(analysis_dir, exist_ok=True)

print(f"\nLoading predictions from: {pred_path}")

df = pd.read_csv(pred_path)
df = df[df["target"] == TARGET].copy()

if df.empty:
    print(f"No data found for target {TARGET}")
    sys.exit(1)


# -----------------------------
# THRESHOLDS
# -----------------------------
thresholds = np.arange(0.4, 0.95, 0.1)
rows = []

print("\nRunning per-instrument multi-threshold analysis...")

for thr in thresholds:

    df["pred"] = (df["y_proba"] >= thr).astype(int)

    for inst, g in df.groupby("Instrument"):

        y_true = g["y_true"]
        y_pred = g["pred"]

        total_rows = len(g)
        signals = int(y_pred.sum())

        # ALWAYS force 2x2 confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        if signals > 0:
            precision = tp / signals
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            fp_rate = fp / signals
        else:
            precision = np.nan
            recall = 0
            fp_rate = np.nan

        rows.append({
            "Instrument": inst,
            "threshold": thr,
            "total_rows": total_rows,
            "signals": signals,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": precision,
            "recall": recall,
            "fp_rate_among_signals": fp_rate,
            "max_proba": g["y_proba"].max(),
            "mean_proba": g["y_proba"].mean(),
            "p95_proba": g["y_proba"].quantile(0.95)
        })


result_df = pd.DataFrame(rows)

out_file = f"{analysis_dir}/instrument_multi_threshold_{MODEL_TYPE}_{TARGET}.csv"
result_df.to_csv(out_file, index=False)

print(f"\nSaved full results to: {out_file}")

# -----------------------------
# PRINT PREVIEW
# -----------------------------
filtered = result_df[result_df["signals"] > 20]

if not filtered.empty:

    print("\n===== TOP 10 BY PRECISION =====")
    print(filtered.sort_values("precision", ascending=False).head(10))

    print("\n===== BOTTOM 10 BY PRECISION =====")
    print(filtered.sort_values("precision", ascending=True).head(10))

else:
    print("\nNo instruments with more than 20 signals for display.")

print("\nDone.")