import pandas as pd
import numpy as np
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python dynamic_threshold_backtest.py <model_type>")
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

df["dynamic_threshold"] = BASE_THRESHOLD + ALPHA * df["vol_z_20"]
df["dynamic_threshold"] = df["dynamic_threshold"].clip(MIN_THR, MAX_THR)

df["pred_dynamic"] = (df["y_proba"] >= df["dynamic_threshold"]).astype(int)

signals = df["pred_dynamic"].sum()
tp = ((df["pred_dynamic"]==1)&(df["y_true"]==1)).sum()
fp = ((df["pred_dynamic"]==1)&(df["y_true"]==0)).sum()

precision = tp / signals if signals>0 else 0

print("Dynamic threshold results:")
print("Signals:", signals)
print("TP:", tp)
print("FP:", fp)
print("Precision:", precision)