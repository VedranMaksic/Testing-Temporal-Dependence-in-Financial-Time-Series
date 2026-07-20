import pandas as pd
import numpy as np
from src.models.config import Config


# ==========================================================
# CORE FUTURE TARGET COMPUTATION
# ==========================================================

def compute_future_targets(df_inst: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """
    Generic future return computation.
    Horizon = broj barova (daily ili hourly, ovisno o timeframe-u).

    Radi:
    - Max future return unutar horizona
    - Min future return unutar horizona
    """

    out = df_inst.sort_index().copy()

    # shift(-1) → gledamo budućnost od sljedećeg bara
    s = out["Close"].shift(-1)

    future_max = (
        s.rolling(horizon, min_periods=horizon)
        .max()
        .shift(-(horizon - 1))
    )

    future_min = (
        s.rolling(horizon, min_periods=horizon)
        .min()
        .shift(-(horizon - 1))
    )

    out[f"MaxRet{horizon}"] = (future_max / out["Close"]) - 1.0
    out[f"MinRet{horizon}"] = (future_min / out["Close"]) - 1.0

    return out


# ==========================================================
# CLASSIFICATION TARGETS
# ==========================================================

def add_classification_targets(df_inst: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Dodaje:
    - UpXX targete
    - DownXX targete
    """

    h = get_effective_horizon(cfg)

    for thr in cfg.up_thresholds:
        df_inst[f"Up{int(thr * 100)}"] = (
            df_inst[f"MaxRet{h}"] >= thr
        ).astype(int)

    for thr in cfg.down_thresholds:
        df_inst[f"Down{int(thr * 100)}"] = (
            df_inst[f"MinRet{h}"] <= -thr
        ).astype(int)

    return df_inst


def get_cls_targets(cfg: Config):
    return (
        [f"Up{int(t * 100)}" for t in cfg.up_thresholds]
        + [f"Down{int(t * 100)}" for t in cfg.down_thresholds]
    )


# ==========================================================
# REGRESSION TARGETS
# ==========================================================

def get_reg_targets(cfg: Config):
    h = get_effective_horizon(cfg)
    return [f"MaxRet{h}", f"MinRet{h}"]


# ==========================================================
# HORIZON LOGIC
# ==========================================================

def get_effective_horizon(cfg: Config) -> int:
    """
    Ako je horizon_steps postavljen → koristi njega
    Inače koristi horizon_days

    Ovo omogućava:
    - daily modele
    - hourly modele
    - bez mijenjanja compute_future_targets logike
    """
    if cfg.horizon_steps is not None:
        return cfg.horizon_steps

    return cfg.horizon_days


# ==========================================================
# GLOBAL TARGET BUILDER
# ==========================================================

def build_targets_global(df_all: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Radi targete po instrumentu.
    """

    h = get_effective_horizon(cfg)

    out_parts = []

    for _, dfi in df_all.groupby("Instrument"):
        dfi = compute_future_targets(dfi, h)
        dfi = add_classification_targets(dfi, cfg)
        out_parts.append(dfi)

    return pd.concat(out_parts).sort_index()