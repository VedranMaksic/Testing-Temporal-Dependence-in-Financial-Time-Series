import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

from src.models.config import Config
from src.models.targets import get_cls_targets, get_reg_targets


try:
    from xgboost import XGBClassifier, XGBRegressor
except Exception as e:
    raise RuntimeError("XGBoost is required. Install with: pip install xgboost") from e


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def time_split(df: pd.DataFrame, cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = df.loc[cfg.train_start : cfg.train_end].copy()
    test = df.loc[cfg.test_start:].copy()
    return train, test


def _make_ohe():
    """
    OneHotEncoder param se razlikuje po verzijama sklearn-a:
    - starije: sparse=
    - novije: sparse_output=
    """
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(cfg: Config) -> ColumnTransformer:
    
    numeric_features = list(cfg.feature_cols)
    categorical_features = ["Instrument"]

    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", _make_ohe(), categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.0,  # forsiraj dense output
    )


def eval_classification(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    met = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None and len(np.unique(y_true)) == 2:
        try:
            met["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            met["roc_auc"] = np.nan
    else:
        met["roc_auc"] = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    met["tn"] = float(cm[0, 0])
    met["fp"] = float(cm[0, 1])
    met["fn"] = float(cm[1, 0])
    met["tp"] = float(cm[1, 1])
    return met


def eval_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))

    # Spearman rank correlation
    try:
        y_true_rank = pd.Series(y_true).rank(method="average").to_numpy()
        y_pred_rank = pd.Series(y_pred).rank(method="average").to_numpy()
        spearman = float(np.corrcoef(y_true_rank, y_pred_rank)[0, 1])
    except Exception:
        spearman = float("nan")

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "spearman": spearman,
    }



def build_xgb_clf(cfg: Config) -> Pipeline:
    pre = build_preprocessor(cfg)
    return Pipeline([
        ("prep", pre),
        ("clf", XGBClassifier(
            n_estimators=600,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=42,
            eval_metric="logloss"
        ))
    ])


def build_xgb_reg(cfg: Config, target: str) -> Pipeline:
    pre = build_preprocessor(cfg)

    # Odabir kvantila po targetu
    if target.startswith("MaxRet"):
        quantile_alpha = 0.9   # gornji rep
    elif target.startswith("MinRet"):
        quantile_alpha = 0.1   # donji rep
    else:
        quantile_alpha = 0.5   # fallback (median)

    return Pipeline([
        ("prep", pre),
        ("reg", XGBRegressor(
            objective="reg:squarederror",  #za mijenjanje cilja sa squarederror na spearmana to jest na kvantil
            #objective= "reg:quantileerror",
            quantile_alpha=quantile_alpha,
            n_estimators=900,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
        ))
    ])



def train_global_classification(df_all: pd.DataFrame, cfg: Config, out_dir: str) -> None:
    ensure_dir(out_dir)

    cls_targets = get_cls_targets(cfg)
    reg_targets = get_reg_targets(cfg)

    needed = list(cfg.feature_cols) + ["Instrument"] + cls_targets + reg_targets
    df = df_all.dropna(subset=needed).copy()

    if len(df) < cfg.min_rows_after_dropna:
        raise RuntimeError(f"Too few rows after dropna: {len(df)}")

    train_df, test_df = time_split(df, cfg)
    if len(train_df) < 500 or len(test_df) < 200:
        raise RuntimeError(f"Too few rows in train/test: train={len(train_df)} test={len(test_df)}")

    X_train = train_df[list(cfg.feature_cols) + ["Instrument"]]
    X_test = test_df[list(cfg.feature_cols) + ["Instrument"]]

    metrics_rows = []
    preds_rows = []

    for target in cls_targets:
        y_train = train_df[target].astype(int).values
        y_test = test_df[target].astype(int).values

        pos_train = int(y_train.sum())
        pos_test = int(y_test.sum())
        if pos_train < cfg.min_pos_in_train or pos_test < cfg.min_pos_in_test:
            continue

        model = build_xgb_clf(cfg)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        met = eval_classification(y_test, y_pred, y_proba)
        met.update({
            "task": "classification",
            "target": target,
            "model": "XGBoost",
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
            "pos_train": pos_train,
            "pos_test": pos_test,
        })
        metrics_rows.append(met)

        out = pd.DataFrame({
            "Date": test_df.index,
            "Instrument": test_df["Instrument"].values,
            "target": target,
            "model": "XGBoost",
            "y_true": y_test,
            "y_pred": y_pred,
            "y_proba": y_proba,
        })
        preds_rows.append(out)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()

    metrics_df.to_csv(os.path.join(out_dir, "classification_metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(out_dir, "classification_predictions.csv"), index=False)


def train_global_regression(df_all: pd.DataFrame, cfg: Config, out_dir: str) -> None:
    ensure_dir(out_dir)

    reg_targets = get_reg_targets(cfg)
    needed = list(cfg.feature_cols) + ["Instrument"] + reg_targets
    df = df_all.dropna(subset=needed).copy()

    train_df, test_df = time_split(df, cfg)
    if len(train_df) < 500 or len(test_df) < 200:
        raise RuntimeError(f"Too few rows in train/test: train={len(train_df)} test={len(test_df)}")

    X_train = train_df[list(cfg.feature_cols) + ["Instrument"]]
    X_test = test_df[list(cfg.feature_cols) + ["Instrument"]]

    metrics_rows = []
    preds_rows = []

    for target in reg_targets:
        y_train = train_df[target].astype(float).values
        y_test = test_df[target].astype(float).values

        model = build_xgb_reg(cfg, target)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        met = eval_regression(y_test, y_pred)
        met.update({
            "task": "regression",
            "target": target,
            "model": "XGBoostReg",
            "n_train": int(len(train_df)),
            "n_test": int(len(test_df)),
        })
        metrics_rows.append(met)

        out = pd.DataFrame({
            "Date": test_df.index,
            "Instrument": test_df["Instrument"].values,
            "target": target,
            "model": "XGBoostReg",
            "y_true": y_test,
            "y_pred": y_pred,
        })
        preds_rows.append(out)

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(preds_rows, ignore_index=True) if preds_rows else pd.DataFrame()

    metrics_df.to_csv(os.path.join(out_dir, "regression_metrics.csv"), index=False)
    preds_df.to_csv(os.path.join(out_dir, "regression_predictions.csv"), index=False)
