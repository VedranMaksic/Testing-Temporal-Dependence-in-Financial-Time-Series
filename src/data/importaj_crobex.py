import matplotlib
matplotlib.use("Agg")   # stabilno na Windowsu (bez Tkintera)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --------------------------------------------------
# PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_EXTERNAL = ROOT / "data" / "external"
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"

DATA_RAW.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

IN_PATH = DATA_EXTERNAL / "crobex_zse.csv"
OUT_PATH = DATA_RAW / "CROBEX.csv"
PLOT_PATH = FIGURES / "CROBEX.png"



def to_float(series: pd.Series) -> pd.Series:
    """
    ZSE koristi decimalni zarez:
    3816,02 -> 3816.02
    """
    return pd.to_numeric(
        series.astype(str)
              .str.replace(".", "", regex=False)
              .str.replace(",", ".", regex=False),
        errors="coerce"
    )

#main

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(
            f"Ne postoji {IN_PATH}. Stavi ZSE export tamo (preimenuj u crobex_zse.csv)."
        )

    df = pd.read_csv(IN_PATH, sep=None, engine="python")

    required = {"date", "open_value", "high_value", "low_value", "last_value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Nedostaju stupci {missing}. Dostupno: {list(df.columns)}")

    out = pd.DataFrame()
    out["Date"] = pd.to_datetime(df["date"], errors="coerce")
    out["Open"] = to_float(df["open_value"])
    out["High"] = to_float(df["high_value"])
    out["Low"] = to_float(df["low_value"])
    out["Close"] = to_float(df["last_value"])

    
    out["Volume"] = (
        to_float(df["turnover"]) if "turnover" in df.columns else 0.0
    )

    out = out.dropna(subset=["Date", "Close"]).sort_values("Date")

    if out["Close"].notna().sum() == 0:
        raise RuntimeError("Close je potpuno prazan nakon konverzije.")

    

    out.to_csv(OUT_PATH, index=False)
    print(f"✅ CROBEX raw spremljen: {OUT_PATH}")
    print(
        f"   range: {out['Date'].min().date()} → {out['Date'].max().date()} "
        f"| rows={len(out)}"
    )

    # nacrtaj

    fig, (ax_price, ax_vol) = plt.subplots(
        2, 1,
        figsize=(12, 6.5),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    ax_price.plot(out["Date"], out["Close"])
    ax_price.set_title("CROBEX (ZSE export)")
    ax_price.set_ylabel("Index points")

    if out["Volume"].notna().any():
        ax_vol.bar(out["Date"], out["Volume"], alpha=0.3)
        ax_vol.set_ylabel("Turnover")

    ax_vol.set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)

    print(f"✅ Plot spremljen: {PLOT_PATH}")

# --------------------------------------------------

if __name__ == "__main__":
    main()
