import warnings
import argparse
from pathlib import Path
from datetime import datetime
import hashlib

import pandas as pd

from src.models.config import Config
from src.models.targets import compute_future_targets, add_classification_targets
from src.models.trainers import (
    ensure_dir,
    train_global_classification,
    train_global_regression,
)
from src.models.report import print_run_summary, ReportConfig

warnings.filterwarnings("ignore")


#ucitaj podatke

def load_features(cfg: Config) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv, parse_dates=["Date"])
    df = df.set_index("Date").sort_index()

    required = {"Close", "Instrument"} | set(cfg.feature_cols)
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {cfg.input_csv}: {missing}")

    for c in ["Close", "Volume", *cfg.feature_cols]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


# napravi target

def build_targets_global(df_all: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    out_parts = []
    for _, dfi in df_all.groupby("Instrument"):
        dfi = compute_future_targets(dfi, cfg.horizon_days)
        dfi = add_classification_targets(dfi, cfg)
        out_parts.append(dfi)

    return pd.concat(out_parts).sort_index()


#main

def main():
    parser = argparse.ArgumentParser(description="Train global ML models")
    parser.add_argument(
        "--feature_set",
        choices=["base", "enhanced"],
        default="enhanced",
        help="Which feature set to use (base or enhanced)",
    )
    args = parser.parse_args()

    cfg = Config(feature_set=args.feature_set)
    ensure_dir(cfg.out_dir)

    
    run_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    feat_hash = hashlib.md5(",".join(cfg.feature_cols).encode("utf-8")).hexdigest()[:8]
    run_name = f"{cfg.feature_set}_{feat_hash}_{run_ts}"

    
    report_path = Path(cfg.out_dir) / f"run_report_{run_name}.txt"

    def log(line: str, fh):
        print(line)
        fh.write(line + "\n")

    
    with open(report_path, "w", encoding="utf-8") as fh:
        log("=" * 60, fh)
        log(f"TRAINING GLOBAL MODEL | FEATURE SET: {cfg.feature_set.upper()}", fh)
        log(f"Run name : {run_name}", fh)
        log(f"Input CSV: {cfg.input_csv}", fh)
        log(f"Output   : {cfg.out_dir}", fh)
        log("=" * 60, fh)
        log("", fh)

        log("TIME SPLIT:", fh)
        log(f"  Train: {cfg.train_start} → {cfg.train_end}", fh)
        log(f"  Test : {cfg.test_start}", fh)
        log("", fh)

        log("TARGETS:", fh)
        log(f"  Horizon days : {cfg.horizon_days}", fh)
        log(f"  Up thresholds: {cfg.up_thresholds}", fh)
        log(f"  Down thresholds: {cfg.down_thresholds}", fh)
        log("", fh)

        log("FEATURES USED (numeric):", fh)
        for f in cfg.feature_cols:
            log(f"  - {f}", fh)
        log("CATEGORICAL (one-hot):", fh)
        log("  - Instrument", fh)
        log("", fh)

        # Load + targets
        df_all = load_features(cfg)
        df_all = build_targets_global(df_all, cfg)

        info_path = Path(cfg.out_dir) / "info.txt"
        with open(info_path, "w", encoding="utf-8") as f:
            f.write(f"Feature set: {cfg.feature_set}\n")
            f.write(f"Input CSV: {cfg.input_csv}\n")
            f.write(f"Horizon: {cfg.horizon_days}\n")
            f.write(f"Up thresholds: {cfg.up_thresholds}\n")
            f.write(f"Down thresholds: {cfg.down_thresholds}\n")
            f.write(f"Train start: {cfg.train_start}\n")
            f.write(f"Train end: {cfg.train_end}\n")
            f.write(f"Test start: {cfg.test_start}\n")
            f.write(f"Features: {list(cfg.feature_cols)}\n")
            f.write("OneHot: Instrument\n")
            f.write("Models: XGBClassifier (Up/Down), XGBRegressor (Max/Min)\n")
            f.write(f"Run name: {run_name}\n")

        # Train + evaluate
        train_global_classification(df_all, cfg, cfg.out_dir)
        train_global_regression(df_all, cfg, cfg.out_dir)

        log("", fh)
        log("✓ DONE", fh)
        log(f"Output folder: {cfg.out_dir}", fh)
        log(" - classification_metrics.csv", fh)
        log(" - classification_predictions.csv", fh)
        log(" - regression_metrics.csv", fh)
        log(" - regression_predictions.csv", fh)
        log("", fh)

        cls_path = Path(cfg.out_dir) / "classification_metrics.csv"
        reg_path = Path(cfg.out_dir) / "regression_metrics.csv"

        if cls_path.exists():
            try:
                cls_df = pd.read_csv(cls_path)
                log("CLASSIFICATION METRICS:", fh)
                log(cls_df.to_string(index=False), fh)
                log("", fh)
            except Exception as e:
                log(f"[WARN] Could not read classification_metrics.csv: {e}", fh)

        if reg_path.exists():
            try:
                reg_df = pd.read_csv(reg_path)
                log("REGRESSION METRICS:", fh)
                log(reg_df.to_string(index=False), fh)
                log("", fh)
            except Exception as e:
                log(f"[WARN] Could not read regression_metrics.csv: {e}", fh)

        log("RUN SUMMARY (console):", fh)
        log("(see console output below)", fh)

    print_run_summary(cfg.out_dir, ReportConfig(decimals=3))
    print(f"\n[OK] Saved per-run report: {report_path}")


if __name__ == "__main__":
    main()
