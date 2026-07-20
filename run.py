import argparse
import shutil
from pathlib import Path
from datetime import datetime
import os

from src.models.config import Config
from src.models.targets import build_targets_global
from src.models.trainers import (
    train_global_classification,
    train_global_regression,
)

# import existing scripts
from src.data.skini_sve import main as download_data
from src.data.nap_df_sa_featureima import main as build_features

import pandas as pd


ROOT = Path(__file__).resolve().parents


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def load_features(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()
    return df


def create_experiment_folder(cfg: Config) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"{ts}_{cfg.feature_set}_{cfg.timeframe}"
    exp_dir = ROOT / "models" / "experiments" / exp_name
    ensure_dir(exp_dir)
    return exp_dir


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_set", type=str, default="enhanced")
    parser.add_argument("--timeframe", type=str, default="1d")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--build", action="store_true")

    args = parser.parse_args()

    cfg = Config(
        feature_set=args.feature_set,
        timeframe=args.timeframe
    )

    # ---------------------------------------
    # Create experiment folder
    # ---------------------------------------
    exp_dir = create_experiment_folder(cfg)
    print(f"\n📁 Experiment folder: {exp_dir}\n")

    # ---------------------------------------
    # 1️⃣ Download data
    # ---------------------------------------
    if args.download:
        print("⬇ Downloading data...")
        download_data()

    # ---------------------------------------
    # 2️⃣ Build features
    # ---------------------------------------
    if args.build:
        print("⚙ Building features...")
        build_features()

    # ---------------------------------------
    # 3️⃣ Load + targets
    # ---------------------------------------
    print("📊 Loading features...")
    df_all = load_features(cfg)

    print("🎯 Building targets...")
    df_all = build_targets_global(df_all, cfg)

    # ---------------------------------------
    # 4️⃣ Train models
    # ---------------------------------------
    print("🤖 Training classification...")
    cls_metrics, cls_preds = train_global_classification(df_all, cfg)

    print("📈 Training regression...")
    reg_metrics, reg_preds = train_global_regression(df_all, cfg)

    # ---------------------------------------
    # 5️⃣ Save everything in experiment folder
    # ---------------------------------------
    cls_metrics.to_csv(exp_dir / "classification_metrics.csv", index=False)
    cls_preds.to_csv(exp_dir / "classification_predictions.csv", index=False)
    reg_metrics.to_csv(exp_dir / "regression_metrics.csv", index=False)
    reg_preds.to_csv(exp_dir / "regression_predictions.csv", index=False)

    print("\n✅ Experiment complete.")
    print(f"Saved in: {exp_dir}")


if __name__ == "__main__":
    main()