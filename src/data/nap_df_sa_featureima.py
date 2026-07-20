import argparse
import importlib
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from src.indicators import indikatori

DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# -load strategy

def load_strategy(strategy_name: str):
    module = importlib.import_module(f"src.strategies.{strategy_name}")
    return module.Strategy()


# load raw

def load_raw(timeframe: str) -> pd.DataFrame:

    raw_dir = DATA_RAW / timeframe

    files = list(raw_dir.glob("*.csv"))

    if len(files) == 0:
        raise FileNotFoundError(f"Nema raw fileova u: {raw_dir}")

    all_dfs = []

    for file in files:

        df = pd.read_csv(file, parse_dates=["Date"])

        instrument = file.stem  # npr AAPL, BTC-USD

        df["Instrument"] = instrument

        all_dfs.append(df)

    df = pd.concat(all_dfs)

    # MultiIndex
    df = df.set_index(["Instrument", "Date"])
    df = df.sort_index()

    # timezone fix (za intraday)
    dates = df.index.get_level_values("Date")

    if hasattr(dates, "tz") and dates.tz is not None:
        df = df.copy()
        df.index = pd.MultiIndex.from_arrays(
            [
                df.index.get_level_values("Instrument"),
                dates.tz_localize(None),
            ],
            names=["Instrument", "Date"]
        )

    # numeric sigurnost
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# apply f

def apply_features(df: pd.DataFrame, strategy) -> pd.DataFrame:

    df = df.copy()

    for feature_name, feature_config in strategy.features.items():

        func_name = feature_config["func"]
        args = feature_config.get("args", [])
        kwargs = feature_config.get("kwargs", {})

        func = getattr(indikatori, func_name)

        df[feature_name] = (
            df.groupby(level="Instrument", group_keys=False)
              .apply(lambda x: func(*(x[arg] for arg in args), **kwargs))
        )

    return df


#build processed

def build_processed(strategy_name: str):

    strategy = load_strategy(strategy_name)

    print(f"\n⚙️ Building processed data")
    print(f"Strategy: {strategy.name}")
    print(f"Timeframe: {strategy.timeframe}")

    df_raw = load_raw(strategy.timeframe)

    # filter po strategiji

    if hasattr(strategy, "instruments") and strategy.instruments:

        print(f"Instruments: {strategy.instruments}")

        df_raw = df_raw.loc[
            df_raw.index.get_level_values("Instrument").isin(strategy.instruments)
        ]

        print(f"Filtered rows: {len(df_raw)}")

    else:
        print("⚠️ No instruments defined → using ALL")

    # feture eng

    df_features = apply_features(df_raw, strategy)

    out_path = DATA_PROCESSED / f"{strategy.name}_{strategy.timeframe}_features.csv"

    df_features.to_csv(out_path)

    print(f"✅ Saved: {out_path}")
    print(f"Rows: {len(df_features)} | Cols: {len(df_features.columns)}\n")


# cli

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()

    build_processed(args.strategy)


if __name__ == "__main__":
    main()