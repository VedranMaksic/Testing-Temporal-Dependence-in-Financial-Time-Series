import sys
import os
import pandas as pd
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


# -----------------------------
# ARGUMENTI IZ KOMANDNE LINIJE
# -----------------------------
if len(sys.argv) != 3:
    print("Usage: python analyze_classification.py <target> <model_type>")
    print("Example: python analyze_classification.py Up10 enhanced")
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
# THRESHOLD ANALYSIS
# -----------------------------
print("\nRunning threshold analysis...")

thresholds = np.arange(0.1, 0.95, 0.05)
results = []

for thr in thresholds:
    y_pred = (df["y_proba"] >= thr).astype(int)
    y_true = df["y_true"]

    if y_pred.sum() == 0:
        continue

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    results.append({
        "threshold": thr,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "signals": int(y_pred.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn
    })

threshold_df = pd.DataFrame(results)
threshold_df.to_csv(f"{analysis_dir}/threshold_analysis_{MODEL_TYPE}_{TARGET}.csv", index=False)

print("\nThreshold results:")
print(threshold_df)

# -----------------------------
# BEST THRESHOLD BY F1
# -----------------------------
best_row = threshold_df.loc[threshold_df["f1"].idxmax()]
best_thr = best_row["threshold"]

print(f"\nBest threshold by F1: {best_thr}")

# -----------------------------
# MONTHLY SIGNAL ANALYSIS
# -----------------------------
print("\nRunning monthly signal analysis...")

df["Date"] = pd.to_datetime(df["Date"])
df["pred"] = (df["y_proba"] >= best_thr).astype(int)

monthly = df[df["pred"] == 1].groupby(df["Date"].dt.to_period("M")).size()
monthly = monthly.rename("signals")

monthly_summary = monthly.describe()
monthly_summary.to_csv(f"{analysis_dir}/monthly_signal_summary_{MODEL_TYPE}_{TARGET}.csv")

print("\nMonthly signal summary:")
print(monthly_summary)

# -----------------------------
# FALSE POSITIVE ANALYSIS
# -----------------------------
print("\nRunning false positive analysis...")

fp_df = df[(df["pred"] == 1) & (df["y_true"] == 0)]
tp_df = df[(df["pred"] == 1) & (df["y_true"] == 1)]

fp_stats = {
    "model_type": MODEL_TYPE,
    "target": TARGET,
    "threshold_used": best_thr,
    "total_signals": int(df["pred"].sum()),
    "true_positives": int(len(tp_df)),
    "false_positives": int(len(fp_df)),
    "precision": float(best_row["precision"]),
    "fp_rate_among_signals": float(len(fp_df) / max(1, df["pred"].sum()))
}

fp_stats_df = pd.DataFrame([fp_stats])
fp_stats_df.to_csv(f"{analysis_dir}/fp_summary_{MODEL_TYPE}_{TARGET}.csv", index=False)

print("\nFalse positive summary:")
print(fp_stats_df)

print(f"\nAnalysis complete.")
print(f"Saved to: {analysis_dir}")