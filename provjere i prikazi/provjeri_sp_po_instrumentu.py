import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

DEFAULT_PATH = "models/ml_output_global_enhanced/regression_predictions.csv"
#DEFAULT_PATH = "models/ml_output_global_base/regression_predictions.csv"

def spearman_corr(y_true, y_pred):
    y_true_rank = pd.Series(y_true).rank(method="average")
    y_pred_rank = pd.Series(y_pred).rank(method="average")
    return float(np.corrcoef(y_true_rank, y_pred_rank)[0, 1])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_path", default=DEFAULT_PATH)
    ap.add_argument("--target", default="MaxRet60_log")  # promijeni po potrebi
    ap.add_argument("--min_n", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_csv(args.pred_path)

    if "target" not in df.columns:
        raise SystemExit(f"[ERROR] 'target' column not found in {args.pred_path}. Columns: {list(df.columns)}")

    df = df[df["target"] == args.target].copy()
    print(f"[INFO] Loaded rows for target={args.target}: {len(df)}")

    if len(df) == 0:
        print("[WARN] No rows after filtering. Available targets:")
        print(df.assign(_t=None))  # just to avoid lint
        all_df = pd.read_csv(args.pred_path)
        print(sorted(all_df["target"].unique()))
        return 0

    rows = []
    for inst, g in df.groupby("Instrument"):
        if len(g) < args.min_n:
            continue

        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()

        rho = spearman_corr(y_true, y_pred)
        r2 = float(r2_score(y_true, y_pred))

        rows.append({
            "Instrument": inst,
            "n_obs": int(len(g)),
            "spearman": rho,
            "r2": r2,
        })

    if not rows:
        print(f"[WARN] No instruments with n_obs >= {args.min_n}. Try lowering --min_n.")
        return 0

    res = pd.DataFrame(rows).sort_values("spearman", ascending=False)

    print("\n=== SUMMARY ===")
    print(res.describe())

    print("\n=== PER INSTRUMENT (sorted by spearman desc) ===")
    print(res)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
