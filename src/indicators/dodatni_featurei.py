from __future__ import annotations
import numpy as np
import pandas as pd


def add_core_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()

    # ---------- Helpers ----------
    def _has(*cols: str) -> bool:
        return all(c in out.columns for c in cols)

    
    for c in ["Close", "SMA_10", "SMA_50", "EMA_20", "RSI_14", "ROC_10", "ATR_14", "OBV"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    # ---------- Trend / MA ----------
    if _has("SMA_10", "SMA_50"):
        out["SMA10_gt_SMA50"] = (out["SMA_10"] > out["SMA_50"]).astype(int)
        
        out["SMA10_cross_SMA50"] = out["SMA10_gt_SMA50"].diff().fillna(0).clip(-1, 1).astype(int)

        out["SPREAD_SMA10_SMA50"] = out["SMA_10"] - out["SMA_50"]

    if _has("Close", "SMA_50"):
        
        out["DIST_Close_SMA50_rel"] = (out["Close"] - out["SMA_50"]) / out["SMA_50"]

    if _has("Close", "EMA_20"):
        out["DIST_Close_EMA20_rel"] = (out["Close"] - out["EMA_20"]) / out["EMA_20"]

    if _has("EMA_20", "SMA_50"):
        out["SPREAD_EMA20_SMA50"] = out["EMA_20"] - out["SMA_50"]

    # ---------- RSI regimes ----------
    if "RSI_14" in out.columns:
        rsi = out["RSI_14"]
        out["RSI_overbought"] = (rsi > 70).astype(int)
        out["RSI_oversold"] = (rsi < 30).astype(int)
        out["RSI_gt_50"] = (rsi > 50).astype(int)
        out["RSI_cross_50"] = out["RSI_gt_50"].diff().fillna(0).clip(-1, 1).astype(int)
        out["RSI_centered"] = rsi - 50

    # ---------- OBV trend + divergence ----------
    
    if _has("OBV", "Close"):
        out["OBV_slope"] = out["OBV"].diff()

        
        price_ch5 = out["Close"].diff(5)
        obv_ch5 = out["OBV"].diff(5)

        out["OBV_div_bear"] = ((price_ch5 > 0) & (obv_ch5 < 0)).astype(int)  # price up, OBV down
        out["OBV_div_bull"] = ((price_ch5 < 0) & (obv_ch5 > 0)).astype(int)  # price down, OBV up

    # ---------- ATR risk context ----------
    if _has("ATR_14", "Close"):
        out["ATR_pct"] = out["ATR_14"] / out["Close"]

    return out


def add_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    
    return add_core_enhanced_features(df)
