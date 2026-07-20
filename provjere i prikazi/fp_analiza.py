import os
import pandas as pd
import numpy as np

PREDICTIONS_PATH = "models/ml_output_global_enhanced/classification_predictions.csv"
FEATURES_PATH = "data/processed/all_instruments_features_enhanced.csv"

TARGET_NAME = "Up10"
THRESHOLD = 0.6


def load_data():
    preds = pd.read_csv(PREDICTIONS_PATH)
    preds = preds[preds["target"] == TARGET_NAME].copy()
    preds["Date"] = pd.to_datetime(preds["Date"])

    features = pd.read_csv(FEATURES_PATH)
    features["Date"] = pd.to_datetime(features["Date"])

    df = preds.merge(
        features,
        on=["Date", "Instrument"],
        how="left"
    )

    df = df.sort_values("Date")
    df = df.set_index("Date")

    return df


def main():

    df = load_data()

    df["signal"] = (df["y_proba"] >= THRESHOLD).astype(int)

    fp = df[(df["signal"] == 1) & (df["y_true"] == 0)]
    tp = df[(df["signal"] == 1) & (df["y_true"] == 1)]

    print("Total signals:", df["signal"].sum())
    print("True positives:", len(tp))
    print("False positives:", len(fp))

    print("\n--- FEATURE MEAN COMPARISON ---\n")

    cols_to_check = [
        "RSI_14",
        "ATR_pct",
        "vol_20",
        "vol_60",
        "trend_strength",
        "mom_20_minus_60",
        "z_close_60"
    ]

    for col in cols_to_check:
        if col in df.columns:
            print(f"\nFeature: {col}")
            print("TP mean:", tp[col].mean())
            print("FP mean:", fp[col].mean())


if __name__ == "__main__":
    main()