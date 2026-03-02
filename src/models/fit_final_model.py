

import argparse
from pathlib import Path
import joblib
import pandas as pd

from src.models.config import Config
from src.models.targets import compute_future_targets, add_classification_targets, get_cls_targets, get_reg_targets
from src.models.trainers import build_xgb_clf, build_xgb_reg, ensure_dir


def load_and_prepare(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    # targets per instrument
    parts = []
    for inst, dfi in df.groupby("Instrument"):
        dfi = compute_future_targets(dfi, cfg.horizon_days)
        dfi = add_classification_targets(dfi, cfg)
        parts.append(dfi)

    df = pd.concat(parts).sort_index()

    needed = list(cfg.feature_cols) + ["Instrument"] + get_cls_targets(cfg) + get_reg_targets(cfg)
    df = df.dropna(subset=needed).copy()

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", choices=["base", "enhanced"], default="enhanced")
    args = parser.parse_args()

    cfg = Config(feature_set=args.feature_set)

    out_dir = Path("models") / f"final_{cfg.feature_set}"
    ensure_dir(out_dir)

    df = load_and_prepare(cfg)

    X = df[list(cfg.feature_cols) + ["Instrument"]]
    #izbaci
    print("feature_set:", cfg.feature_set)
    print("input_csv:", cfg.input_csv)
    print("n_features:", len(cfg.feature_cols), cfg.feature_cols)
    #do tud
    # === CLASSIFICATION ===
    for target in get_cls_targets(cfg):
        y = df[target].astype(int).values
        model = build_xgb_clf(cfg)
        model.fit(X, y)
        joblib.dump(model, out_dir / f"clf_{target}.joblib")

    # === REGRESSION ===
    for target in get_reg_targets(cfg):
        y = df[target].astype(float).values
        model = build_xgb_reg(cfg)
        model.fit(X, y)
        joblib.dump(model, out_dir / f"reg_{target}.joblib")

    print(f"✅ FINAL MODELS TRAINED → {out_dir}")


if __name__ == "__main__":
    main()
