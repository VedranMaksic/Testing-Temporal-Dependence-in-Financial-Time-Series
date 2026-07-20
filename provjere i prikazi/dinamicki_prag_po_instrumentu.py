import pandas as pd
import numpy as np
import sys
import os
from sklearn.metrics import confusion_matrix

if len(sys.argv) != 2:
    print("Usage: python dynamic_threshold_by_instrument.py <model_type>")
    sys.exit(1)

MODEL_TYPE = sys.argv[1]

BASE_THRESHOLD = 0.6
ALPHA = 0.05
MIN_THR = 0.4
MAX_THR = 0.8

pred_path = f"models/ml_output_global_{MODEL_TYPE}/classification_predictions.csv"
feature_path = f"data/processed/all_instruments_features_enhanced.csv"

df_pred = pd.read_csv(pred_path)
df_feat = pd.read_csv(feature_path)

df = df_pred[df_pred["target"] == "Up10"].copy()

df["Date"] = pd.to_datetime(df["Date"])
df_feat["Date"] = pd.to_datetime(df_feat["Date"])

df = df.merge(
    df_feat[["Date","Instrument","vol_z_20"]],
    on=["Date","Instrument"],
    how="left"
)

# dynamic threshold
df["dynamic_threshold"] = BASE_THRESHOLD + ALPHA * df["vol_z_20"]
df["dynamic_threshold"] = df["dynamic_threshold"].clip(MIN_THR, MAX_THR)

df["pred_dynamic"] = (df["y_proba"] >= df["dynamic_threshold"]).astype(int)

rows = []

for inst, g in df.groupby("Instrument"):

    y_true = g["y_true"]
    y_pred = g["pred_dynamic"]

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    tn, fp, fn, tp = cm.ravel()

    signals = int(y_pred.sum())

    precision = tp / signals if signals > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    fp_rate = fp / signals if signals > 0 else np.nan

    rows.append({
        "Instrument": inst,
        "signals": signals,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "fp_rate_among_signals": fp_rate,
        "max_proba": g["y_proba"].max(),
        "mean_proba": g["y_proba"].mean()
    })

result = pd.DataFrame(rows).sort_values("precision", ascending=False)

print("\n===== Dynamic Threshold Results by Instrument =====\n")
print(result)

# spremi
out_dir = f"models/ml_output_global_{MODEL_TYPE}/analysis_dynamic"
os.makedirs(out_dir, exist_ok=True)

result.to_csv(f"{out_dir}/dynamic_threshold_by_instrument.csv", index=False)

print("\nSaved to:", out_dir)