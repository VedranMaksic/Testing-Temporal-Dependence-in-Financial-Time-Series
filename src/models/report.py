# src/models/report.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ReportConfig:
    # koliko decimala ispisivati
    decimals: int = 3
    # ako ima više modela po targetu, po čemu biramo "najbolji"
    cls_sort_by: str = "f1"   # ili "roc_auc"
    reg_sort_by: str = "r2"


def _read_csv_if_exists(path: str) -> Optional[pd.DataFrame]:
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def print_classification_summary(out_dir: str, cfg: ReportConfig = ReportConfig()) -> None:
    """
    Čita models/.../classification_metrics.csv i ispisuje sažetak po targetu (Up10, Down10).
    """
    path = os.path.join(out_dir, "classification_metrics.csv")
    m = _read_csv_if_exists(path)
    if m is None:
        print("⚠️  No classification metrics found:", path)
        return

    # očekujemo ove stupce, ali ne rušimo se ako ih nema
    want = ["target", "model", "f1", "precision", "recall", "roc_auc", "accuracy", "tp", "fp", "fn", "tn"]
    cols = [c for c in want if c in m.columns]
    m = m[cols].copy()

    print("\n=== CLASSIFICATION SUMMARY (test period) ===")
    for tgt in m["target"].unique():
        mt = m[m["target"] == tgt].copy()

        sort_col = cfg.cls_sort_by if cfg.cls_sort_by in mt.columns else None
        if sort_col:
            mt = mt.sort_values(sort_col, ascending=False)

        best = mt.iloc[0].to_dict()
        d = cfg.decimals

        def fmt(x):
            try:
                return f"{float(x):.{d}f}"
            except Exception:
                return "nan"

        line = (
            f"{tgt} | {best.get('model', '?')} | "
            f"F1={fmt(best.get('f1'))} "
            f"P={fmt(best.get('precision'))} "
            f"R={fmt(best.get('recall'))} "
            f"AUC={fmt(best.get('roc_auc'))} "
            f"Acc={fmt(best.get('accuracy'))}"
        )
        print(line)

        if all(k in best for k in ["tp", "fp", "fn", "tn"]):
            try:
                print(f"   CM: TP={int(best['tp'])} FP={int(best['fp'])} FN={int(best['fn'])} TN={int(best['tn'])}")
            except Exception:
                pass


def print_regression_summary(out_dir: str, cfg: ReportConfig = ReportConfig()) -> None:
    """
    Čita models/.../regression_metrics.csv i ispisuje sažetak po targetu (MaxRet60, MinRet60).
    """
    path = os.path.join(out_dir, "regression_metrics.csv")
    m = _read_csv_if_exists(path)
    if m is None:
        print("⚠️  No regression metrics found:", path)
        return

    want = ["target", "model", "r2", "mae", "rmse"]
    cols = [c for c in want if c in m.columns]
    m = m[cols].copy()

    print("\n=== REGRESSION SUMMARY (test period) ===")
    for tgt in m["target"].unique():
        mt = m[m["target"] == tgt].copy()

        sort_col = cfg.reg_sort_by if cfg.reg_sort_by in mt.columns else None
        if sort_col:
            mt = mt.sort_values(sort_col, ascending=False)

        best = mt.iloc[0].to_dict()
        d = cfg.decimals

        def fmt(x, dec=d):
            try:
                return f"{float(x):.{dec}f}"
            except Exception:
                return "nan"

        print(
            f"{tgt} | {best.get('model','?')} | "
            f"R2={fmt(best.get('r2'))} "
            f"MAE={fmt(best.get('mae'), 4)} "
            f"RMSE={fmt(best.get('rmse'), 4)}"
        )


def print_run_summary(out_dir: str, cfg: ReportConfig = ReportConfig()) -> None:
    """
    Convenience: ispiše i klasifikaciju i regresiju.
    """
    print_classification_summary(out_dir, cfg)
    print_regression_summary(out_dir, cfg)
