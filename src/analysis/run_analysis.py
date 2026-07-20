import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_score, recall_score

import pandas as pd
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score
)

def analyze(pred_path, threshold):

    df = pd.read_csv(pred_path, parse_dates=["Date"])

    df["y_pred"] = (df["y_proba"] >= threshold).astype(int)



    print("\n===== GLOBAL METRICS =====")
    print("Precision:", precision_score(df["y_true"], df["y_pred"]))
    print("Recall:", recall_score(df["y_true"], df["y_pred"]))
    print("ROC AUC:", roc_auc_score(df["y_true"], df["y_proba"]))


    print("\nInstruments in prediction file:")
    print(df["Instrument"].unique())

    rows = []

    for instrument, group in df.groupby("Instrument"):

        cm = confusion_matrix(group["y_true"], group["y_pred"])

        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn = fp = fn = tp = 0

        precision = precision_score(group["y_true"], group["y_pred"], zero_division=0)
        recall = recall_score(group["y_true"], group["y_pred"], zero_division=0)

        try:
            roc_auc = roc_auc_score(group["y_true"], group["y_proba"])
        except:
            roc_auc = np.nan

        signals = group["y_pred"].sum()
        real_events = group["y_true"].sum()

        monthly = (
            group[group["y_pred"] == 1]
            .groupby(group["Date"].dt.to_period("M"))
            .size()
        )

        avg_monthly = monthly.mean() if len(monthly) > 0 else 0

        rows.append({
            "Instrument": instrument,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
            "Precision": round(precision, 3),
            "Recall": round(recall, 3),
            "ROC_AUC": round(roc_auc, 3),
            "Signals": signals,
            "True_Events": real_events,
            "Avg_Signals_Month": round(avg_monthly, 2)
        })

    table = pd.DataFrame(rows)
    print("\n===== PER INSTRUMENT TABLE =====\n")
    print(table.to_string(index=False))

def threshold_sweep(pred_path):

    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

    for t in thresholds:
        print(f"\n======================")
        print(f"THRESHOLD = {t}")
        print(f"======================")
        analyze(pred_path, t)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--sweep", action="store_true")

    args = parser.parse_args()

    if args.sweep:
        threshold_sweep(args.predictions)
    else:
        analyze(args.predictions, args.threshold)


if __name__ == "__main__":
    main()