import os
import pandas as pd
import numpy as np

PREDICTIONS_PATH = "models/ml_output_global_enhanced/classification_predictions.csv"
OUTPUT_DIR = "models/ml_output_global_enhanced/analysis"

TARGET_NAME = "Up10"


THRESHOLDS = [0.4, 0.5, 0.6, 0.7]


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_data():
    df = pd.read_csv(PREDICTIONS_PATH)
    df = df[df["target"] == TARGET_NAME].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def compute_stats(df):

    rows = []

    instruments = df["Instrument"].unique()

    for instrument in instruments:

        g = df[df["Instrument"] == instrument].copy()

        # distribucija vjerojatnosti (neovisno o pragu)
        max_proba = g["y_proba"].max()
        mean_proba = g["y_proba"].mean()
        p95_proba = g["y_proba"].quantile(0.95)

        for thr in THRESHOLDS:

            g["signal"] = (g["y_proba"] >= thr).astype(int)

            total_signals = int(g["signal"].sum())

            tp = int(((g["signal"] == 1) & (g["y_true"] == 1)).sum())
            fp = int(((g["signal"] == 1) & (g["y_true"] == 0)).sum())
            fn = int(((g["signal"] == 0) & (g["y_true"] == 1)).sum())
            tn = int(((g["signal"] == 0) & (g["y_true"] == 0)).sum())

            precision = tp / total_signals if total_signals > 0 else np.nan
            recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan

            rows.append({
                "Instrument": instrument,
                "threshold": thr,
                "total_rows": len(g),
                "signals": total_signals,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "precision": precision,
                "recall": recall,
                "max_proba": max_proba,
                "mean_proba": mean_proba,
                "p95_proba": p95_proba
            })

    result = pd.DataFrame(rows)

    return result


def main():

    ensure_dir(OUTPUT_DIR)

    df = load_data()

    stats = compute_stats(df)

    stats.to_csv(
        os.path.join(OUTPUT_DIR, "instrument_multi_threshold_stats.csv"),
        index=False
    )

    print("\nSaved to:", OUTPUT_DIR)

    print("\nPreview:")
    print(stats.sort_values(["Instrument", "threshold"]).head(20))


if __name__ == "__main__":
    main()