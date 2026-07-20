from __future__ import annotations
import numpy as np
import pandas as pd


def add_more_features(df: pd.DataFrame, lag: int = 1) -> pd.DataFrame:
   
    out = df.copy()

    
    for c in ["Close", "Volume", "SMA_10", "SMA_50", "EMA_20", "RSI_14", "ROC_10", "ATR_14", "OBV", "ATR_pct"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    
    if "Date" in out.columns:
        out = out.sort_values(["Instrument", "Date"]).copy()
    else:
        
        out = out.sort_values(["Instrument"] + ([] if out.index.name is None else [out.index.name])).copy()

    g = out.groupby("Instrument", sort=False)

    new_cols = []

    
    # Close kontekst
    
    if "Close" in out.columns:
        out["log_close"] = np.log(out["Close"].where(out["Close"] > 0))
        new_cols += ["log_close"]

        # z-score Close (60)
        m60 = g["Close"].transform(lambda s: s.rolling(60, min_periods=60).mean())
        sd60 = g["Close"].transform(lambda s: s.rolling(60, min_periods=60).std())
        out["z_close_60"] = (out["Close"] - m60) / sd60
        new_cols += ["z_close_60"]

        # distance to recent high/low (60)
        rmax60 = g["Close"].transform(lambda s: s.rolling(60, min_periods=60).max())
        rmin60 = g["Close"].transform(lambda s: s.rolling(60, min_periods=60).min())
        out["dist_high_60"] = (out["Close"] / rmax60) - 1.0
        out["dist_low_60"] = (out["Close"] / rmin60) - 1.0
        new_cols += ["dist_high_60", "dist_low_60"]

    
    # volatilnost
    
    if "Close" in out.columns:
        # log returni
        out["ret_1d"] = g["Close"].transform(lambda s: np.log(s).diff())
        out["ret_5d"] = g["Close"].transform(lambda s: np.log(s).diff(5))
        out["ret_20d"] = g["Close"].transform(lambda s: np.log(s).diff(20))
        new_cols += ["ret_1d", "ret_5d", "ret_20d"]

        # rolling vol (std) nad ret_1d
        out["vol_20"] = g["ret_1d"].transform(lambda s: s.rolling(20, min_periods=20).std())
        out["vol_60"] = g["ret_1d"].transform(lambda s: s.rolling(60, min_periods=60).std())
        out["vol_ratio_20_60"] = out["vol_20"] / out["vol_60"]
        new_cols += ["vol_20", "vol_60", "vol_ratio_20_60"]

    if "ATR_pct" in out.columns:
        # ATR_pct z-score (60)
        m60 = g["ATR_pct"].transform(lambda s: s.rolling(60, min_periods=60).mean())
        sd60 = g["ATR_pct"].transform(lambda s: s.rolling(60, min_periods=60).std())
        out["atr_pct_z_60"] = (out["ATR_pct"] - m60) / sd60
        new_cols += ["atr_pct_z_60"]

    
    #momentum 
    
    if "ret_20d" in out.columns and "ret_1d" in out.columns:
        # ret_60d
        out["ret_60d"] = g["Close"].transform(lambda s: np.log(s).diff(60))
        out["mom_20_minus_60"] = out["ret_20d"] - out["ret_60d"]
        new_cols += ["ret_60d", "mom_20_minus_60"]

    if all(c in out.columns for c in ["EMA_20", "SMA_50", "ATR_14"]):
        # trend strength normaliziran ATR-om
        out["trend_strength"] = (out["EMA_20"] - out["SMA_50"]).abs() / out["ATR_14"].replace(0, np.nan)
        new_cols += ["trend_strength"]

    # rsi jos stvari
    if "RSI_14" in out.columns:
        out["RSI_abs_dist_50"] = (out["RSI_14"] - 50).abs()
        out["RSI_regime_3"] = np.select(
            [out["RSI_14"] < 30, out["RSI_14"] > 70],
            [0, 2],
            default=1
        ).astype(float)
        new_cols += ["RSI_abs_dist_50", "RSI_regime_3"]

        # EMA na RSI (zaglađivanje)
        out["RSI_ema_10"] = g["RSI_14"].transform(lambda s: s.ewm(span=10, adjust=False).mean())
        new_cols += ["RSI_ema_10"]

    # volumen
    if "Volume" in out.columns:
        vmean20 = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).mean())
        vstd20 = g["Volume"].transform(lambda s: s.rolling(20, min_periods=20).std())
        out["vol_z_20"] = (out["Volume"] - vmean20) / vstd20
        out["vol_ratio_5_20"] = (
            g["Volume"].transform(lambda s: s.rolling(5, min_periods=5).mean()) / vmean20
        )
        new_cols += ["vol_z_20", "vol_ratio_5_20"]

    if "OBV" in out.columns:
        out["obv_slope_1"] = g["OBV"].transform(lambda s: s.diff())
        out["obv_slope_5"] = g["OBV"].transform(lambda s: s.diff(5))
        new_cols += ["obv_slope_1", "obv_slope_5"]

        if "ATR_14" in out.columns:
            out["obv_slope_1_over_atr"] = out["obv_slope_1"] / out["ATR_14"].replace(0, np.nan)
            new_cols += ["obv_slope_1_over_atr"]

    # MA
    if "Close" in out.columns and "SMA_50" in out.columns:
        out["Close_above_SMA50"] = (out["Close"] > out["SMA_50"]).astype(float)
        new_cols += ["Close_above_SMA50"]

    if "Close" in out.columns and "EMA_20" in out.columns:
        out["Close_above_EMA20"] = (out["Close"] > out["EMA_20"]).astype(float)
        new_cols += ["Close_above_EMA20"]

    if "SMA_50" in out.columns:
        out["slope_SMA50_10"] = g["SMA_50"].transform(lambda s: s.diff(10))
        new_cols += ["slope_SMA50_10"]

   
    if lag and lag > 0 and len(new_cols) > 0:
        out[new_cols] = g[new_cols].shift(lag)

    return out
