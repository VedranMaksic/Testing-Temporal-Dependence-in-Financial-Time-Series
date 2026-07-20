import argparse
import pandas as pd
from pathlib import Path
import yfinance as yf

ROOT = Path(__file__).resolve().parents[2]


INSTRUMENTS = {

    # =========================
    # CRYPTO
    # =========================
    "Bitcoin": "BTC-USD",
    "Ethereum": "ETH-USD",
    "Binance Coin": "BNB-USD",
    "Solana": "SOL-USD",
    "XRP": "XRP-USD",
    "Cardano": "ADA-USD",
    "Dogecoin": "DOGE-USD",

    # =========================
    # METALS
    # =========================
    "Gold": "GC=F",
    "Silver": "SI=F",
    "Copper": "HG=F",

    # =========================
    # BONDS
    # =========================
    "US 7-10Y Treasuries": "IEF",
    "US 20Y+ Treasuries": "TLT",
    "US Short-Term Treasuries": "SHY",

    # =========================
    # STOCKS (TECH / BIG CAP)
    # =========================
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Meta": "META",
    "Nvidia": "NVDA",
    "Tesla": "TSLA",

    # =========================
    # INDICES
    # =========================
    "S&P 500": "^GSPC",
    "Nasdaq 100": "^NDX",
    "Dow Jones": "^DJI",
    "Russell 2000": "^RUT",

    # =========================
    # FX
    # =========================
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
}

START_DATE = "1995-01-01"


def download_one(ticker: str, timeframe: str) -> pd.DataFrame:

    if timeframe == "1d":
        df = yf.download(
            ticker,
            start=START_DATE,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
    else:
        # minimalna promjena: period ovisno o timeframeu
        if timeframe == "1m":
            period = "7d"
        elif timeframe in ["2m", "5m", "15m", "30m"]:
            period = "60d"
        elif timeframe in ["60m", "1h"]:
            period = "730d"
        else:
            period = "60d"  # fallback

        df = yf.download(
            ticker,
            period=period,
            interval=timeframe,
            progress=False,
            auto_adjust=False,
        )

    if df is None or df.empty:
        return pd.DataFrame()

    # Ako Yahoo vrati MultiIndex stupce
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    keep_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]]

    
    df = df.reset_index()
    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)
    df = df.set_index("Date")

    df = df.sort_index()

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--timeframe",
        default="1d",
        help="Timeframe: 1d, 1h, 30m, 15m..."
    )
    args = parser.parse_args()
    timeframe = args.timeframe

    raw_dir = ROOT / "data" / "raw" / timeframe
    raw_dir.mkdir(parents=True, exist_ok=True)

    merged_parts = []

    print(f"\nDownloading data for timeframe: {timeframe}\n")

    for name, ticker in INSTRUMENTS.items():
        df = download_one(ticker, timeframe)

        if df.empty:
            print(f"{name} ✗")
            continue

        df = df.copy()
        df["Instrument"] = name

        # spremanje pojedinačnog instrumenta
        out_path = raw_dir / f"{ticker}.csv"
        df.to_csv(out_path, index_label="Date")

        merged_parts.append(df)
        print(f"{name} ✓")

    if not merged_parts:
        raise RuntimeError("No instruments downloaded.")

    # Spajanje svih instrumenata
    all_raw = pd.concat(merged_parts)

    # Ključno: sortiranje po Instrument + Date
    all_raw = all_raw.sort_values(["Instrument", all_raw.index.name or "Date"])

    out_path = raw_dir / "all_instruments_raw.csv"
    all_raw.to_csv(out_path, index_label="Date")

    print(f"\nALL ✓ saved to data/raw/{timeframe}/all_instruments_raw.csv")





if __name__ == "__main__":
    main()