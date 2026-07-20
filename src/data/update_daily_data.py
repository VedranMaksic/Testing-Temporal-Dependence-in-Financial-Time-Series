import pandas as pd
import yfinance as yf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "raw" / "1d"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ============================================
# CLEAN DF
# ============================================

def clean_df(df):

    df = df.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Date", "Open", "High", "Low", "Close", "Volume"]]

    return df


# ============================================
# UPDATE ONE ASSET
# ============================================

def update_asset(ticker):

    path = DATA_DIR / f"{ticker}.csv"

    if path.exists():

        df_old = pd.read_csv(path, parse_dates=["Date"])
        last_date = df_old["Date"].max()

        print(f"{ticker}: updating from {last_date}")

        df_new = yf.download(
            ticker,
            start=last_date + pd.Timedelta(days=1),
            auto_adjust=False,
            progress=False
        )

        if not df_new.empty:
            df_new = clean_df(df_new)

            df = pd.concat([df_old, df_new])
            df = df.drop_duplicates("Date")
        else:
            print(f"{ticker}: no new data")
            df = df_old

    else:

        print(f"{ticker}: downloading full history")

        df = yf.download(
            ticker,
            period="5y",
            interval="1d",
            auto_adjust=False,
            progress=False
        )

        df = clean_df(df)

    df.to_csv(path, index=False)


# ============================================
# MAIN
# ============================================

def main(tickers=None):

    print("🚀 Updating daily data...\n")

    # ============================================
    # ako nisu zadani tickers → koristi postojeće CSV-ove
    # ============================================

    if tickers is None:

        files = list(DATA_DIR.glob("*.csv"))

        if not files:
            print("❌ No existing data found.")
            return

        tickers = [f.stem for f in files]

        print("Using existing tickers:")
        print(tickers)

    else:
        print("Using provided tickers:")
        print(tickers)

    # ============================================
    # UPDATE LOOP
    # ============================================

    for ticker in tickers:
        try:
            update_asset(ticker)
        except Exception as e:
            print(f"❌ Error for {ticker}: {e}")

    print("\n✅ Data updated")


if __name__ == "__main__":
    main()