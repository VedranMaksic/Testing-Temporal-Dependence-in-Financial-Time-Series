

import argparse
from pathlib import Path
from datetime import datetime
import joblib
import pandas as pd

from src.models.config import Config
from src.models.targets import get_cls_targets, get_reg_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", choices=["base", "enhanced"], default="enhanced")
    parser.add_argument("--last_n", type=int, default=1)
    args = parser.parse_args()

    cfg = Config(feature_set=args.feature_set)
    model_dir = Path("models") / f"final_{cfg.feature_set}"

    df = pd.read_csv(cfg.input_csv, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    # zadnjih N redova po instrumentu
    df_pred = df.groupby("Instrument", group_keys=False).tail(args.last_n).copy()

    X = df_pred[list(cfg.feature_cols) + ["Instrument"]]

    out = df_pred[["Instrument", "Close"]].copy()

    # === CLASSIFICATION ===
    for target in get_cls_targets(cfg):
        model = joblib.load(model_dir / f"clf_{target}.joblib")
        out[f"P_{target}"] = model.predict_proba(X)[:, 1]

    # === REGRESSION ===
    for target in get_reg_targets(cfg):
        model = joblib.load(model_dir / f"reg_{target}.joblib")
        out[f"{target}_pred"] = model.predict(X)

    ts = datetime.now().strftime("%Y-%m-%d_%H%M")
    out_path = Path("models") / f"predictions_live_{cfg.feature_set}_{ts}.csv"
    out.to_csv(out_path, index_label="Date")

    print(f"✅ PREDICTIONS SAVED → {out_path}")
    print("Rows:", len(out))


if __name__ == "__main__":
    main()
