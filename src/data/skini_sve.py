import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import matplotlib
matplotlib.use("Agg")  # stabilno na Windowsu (bez Tkintera)

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import yfinance as yf




ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = ROOT / "data" / "raw"
FIGURES = ROOT / "reports" / "figures"

DATA_RAW.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# INSTRUMENTI (CROBEX se NE skida - dolazi iz ZSE)
# --------------------------------------------------

INSTRUMENTS_YAHOO = {
    "S&P 500": ("^GSPC", True, "Index points"),
    "EUR/USD": ("EURUSD=X", False, "USD per EUR"),
    "Apple": ("AAPL", True, "USD"),
    "Bitcoin": ("BTC-USD", True, "USD"),
    "Gold": ("GC=F", True, "USD/oz"),
    "US Treasuries (IEF)": ("IEF", True, "USD"),
    "Germany 10Y": ("IEGA.L", True, "EUR"),
}

START_DATE = "1995-01-01"
CROBEX_LOCAL_PATH = DATA_RAW / "CROBEX.csv"



def safe_name(t: str) -> str:
    return t.replace("^", "_").replace("/", "-").replace("=", "_").replace(".", "_")


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        level0 = df.columns.get_level_values(0)
        if len(pd.unique(level0)) == df.shape[1]:
            df.columns = level0
        else:
            df.columns = [
                "_".join([str(x) for x in tup if str(x) != ""])
                for tup in df.columns.to_flat_index()
            ]
    return df


def pick_price_col(df: pd.DataFrame) -> str:
    if "Close" in df.columns:
        return "Close"
    if "Adj Close" in df.columns:
        return "Adj Close"
    return df.columns[0]


def fetch_to_csv_yahoo(ticker: str) -> tuple[pd.DataFrame, Path]:
    df = yf.download(
        ticker,
        start=START_DATE,
        progress=False,
        group_by="column",
        auto_adjust=False,   #  uklanja FutureWarning i drži output stabilnim
    )

    if df is None or df.empty:
        return pd.DataFrame(), Path()

    df = flatten_columns(df)

    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    if keep:
        df = df[keep]

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_index()

    out_path = DATA_RAW / f"{safe_name(ticker)}.csv"
    df.to_csv(out_path, index_label="Date")
    return df, out_path


def load_local_crobex(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.dropna(subset=["Date"]).sort_values("Date").set_index("Date")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def plot_price_and_volume(name: str, file_stem: str, df: pd.DataFrame, draw_volume: bool, y_unit: str) -> Path:
    price_col = pick_price_col(df)
    png_path = FIGURES / f"{safe_name(file_stem)}.png"

    if draw_volume and ("Volume" in df.columns) and df["Volume"].notna().any():
        fig, (ax_price, ax_vol) = plt.subplots(
            2, 1, figsize=(12, 6.5), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]}
        )

        ax_price.plot(df.index, df[price_col])
        ax_price.set_title(f"{name} – {price_col}")
        ax_price.set_ylabel(y_unit)

        ax_vol.bar(df.index, df["Volume"], alpha=0.3)
        ax_vol.set_ylabel("Volume")
        ax_vol.set_xlabel("Date")

        fig.tight_layout()
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    else:
        plt.figure(figsize=(12, 5))
        plt.plot(df.index, df[price_col])
        plt.title(f"{name} – {price_col}")
        plt.ylabel(y_unit)
        plt.xlabel("Date")
        plt.tight_layout()
        plt.savefig(png_path, dpi=150)
        plt.close()

    return png_path

#main

def main():
    merged_parts = []

    # Yahoo instrumenti
    for name, (ticker, draw_vol, unit) in INSTRUMENTS_YAHOO.items():
        df, csv_path = fetch_to_csv_yahoo(ticker)
        if df.empty:
            print(f"{name} ✗")
            continue

        png_path = plot_price_and_volume(name, ticker, df, draw_vol, unit)

        df2 = df.copy()
        df2["Instrument"] = name
        merged_parts.append(df2)

        print(f"{name} ✓  {csv_path}  |  {png_path}")

    # CROBEX lokalno
    if CROBEX_LOCAL_PATH.exists():
        cro = load_local_crobex(CROBEX_LOCAL_PATH)
        if not cro.empty:
            png_path = plot_price_and_volume("CROBEX", "CROBEX", cro, True, "Index points")

            cro2 = cro.copy()
            cro2["Instrument"] = "CROBEX"
            merged_parts.append(cro2)

            print(f"CROBEX ✓  {CROBEX_LOCAL_PATH}  |  {png_path}")
        else:
            print("CROBEX ✗")
    else:
        print("CROBEX ✗")

    if not merged_parts:
        raise RuntimeError("Nijedan instrument nije uspješno pripremljen.")

    all_raw = pd.concat(merged_parts).sort_index()
    out_path = DATA_RAW / "all_instruments_raw.csv"
    all_raw.to_csv(out_path, index_label="Date")

    print(f"ALL ✓  {out_path}")


if __name__ == "__main__":
    main()
