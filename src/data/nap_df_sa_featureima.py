import sys
from pathlib import Path

from src.indicators.jos_dodatnih_feturea import add_more_features

ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "src"))

import pandas as pd

from indicators.indikatori import sma, ema, roc, rsi_wilder, atr_wilder, obv
from indicators.dodatni_featurei import add_enhanced_features

RAW_PATH = ROOT / "data" / "raw" / "all_instruments_raw.csv"

OUT_DIR = ROOT / "data" / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_BASE = OUT_DIR / "all_instruments_features_base.csv"
OUT_ENH = OUT_DIR / "all_instruments_features_enhanced.csv"


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").set_index("Date")

    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def add_base_features_one_instrument(df: pd.DataFrame) -> pd.DataFrame:
    """
    Base: samo indikatori (bez dodatnih izvedenih featurea)
    df: DataFrame za jedan instrument (index=Date)
    """
    df = df.sort_index().copy()

    # trend
    df["SMA_10"] = sma(df["Close"], 10)
    df["SMA_50"] = sma(df["Close"], 50)
    df["EMA_20"] = ema(df["Close"], 20)

    # momentum
    df["RSI_14"] = rsi_wilder(df["Close"], 14)
    df["ROC_10"] = roc(df["Close"], 10)

    # volatilnost
    df["ATR_14"] = atr_wilder(df["High"], df["Low"], df["Close"], 14)

    # volumen
    df["OBV"] = obv(df["Close"], df["Volume"])

    return df


def main():
    if not RAW_PATH.exists():
        raise FileNotFoundError(
            f"Nema RAW filea: {RAW_PATH}. Prvo pokreni src/data/skini_sve.py"
        )

    df_all = load_raw(RAW_PATH)

    if "Instrument" not in df_all.columns:
        raise ValueError("RAW dataset nema stupac 'Instrument'.")

    out_base_parts = []
    out_enh_parts = []

    for instr, df_instr in df_all.groupby("Instrument"):
        # 1) base
        df_base = add_base_features_one_instrument(df_instr)

        # 2) enhanced = base + dodatni featurei
        df_enh = add_enhanced_features(df_base.copy())

        #3) ovdje se dodaju svi dodatni featurei
        df_enh = add_more_features(df_enh, lag=1) 

        out_base_parts.append(df_base)
        out_enh_parts.append(df_enh)

    df_base_all = pd.concat(out_base_parts).sort_index()
    df_enh_all = pd.concat(out_enh_parts).sort_index()

    # makni redove gdje indikatori nisu definirani
    drop_subset = ["SMA_50", "RSI_14", "ATR_14"]
    df_base_all = df_base_all.dropna(subset=drop_subset)
    df_enh_all = df_enh_all.dropna(subset=drop_subset)

    df_base_all.to_csv(OUT_BASE, index_label="Date")
    df_enh_all.to_csv(OUT_ENH, index_label="Date")

    print(f"✅ BASE features spremljeni: {OUT_BASE} | rows={len(df_base_all)} | cols={len(df_base_all.columns)}")
    print(f"✅ ENH  features spremljeni: {OUT_ENH}  | rows={len(df_enh_all)}  | cols={len(df_enh_all.columns)}")


if __name__ == "__main__":
    main()
