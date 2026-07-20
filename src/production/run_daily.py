import pandas as pd
from pathlib import Path
import importlib

from src.data.update_daily_data import main as update_data
from src.data.nap_df_sa_featureima import build_processed

from src.production.predictor import Predictor
from src.production.signals import generate_signals
from src.production.notifier import send_email

ROOT = Path(__file__).resolve().parents[2]

STRATEGIES = [
    "up5_high_vol",
    "up5_mid_vol",
    "up5_low_vol"
]

THRESHOLD = 0.3
MAX_SIGNALS = 8


def load_strategy(strategy_name):
    module = importlib.import_module(f"src.strategies.{strategy_name}")
    return module.Strategy()


def format_message(all_signals):

    if not all_signals:
        return "No signals today."

    msg = "DAILY SIGNALS\n\n"

    for strategy_name, signals in all_signals.items():

        msg += f"=== {strategy_name} ===\n"

        if signals.empty:
            msg += "No signals\n\n"
            continue

        for _, row in signals.iterrows():
            msg += f"{row['Instrument']} | prob={row['prob']:.2f}\n"

        msg += "\n"

    return msg


def main():

    print("\n🚀 MULTI-STRATEGY RUN\n")

    # 1. UPDATE DATA (jednom!)
    print("📥 Updating raw data...")
    update_data()

    all_signals = {}

    # =========================
    # LOOP kroz strategije
    # =========================

    for strategy_name in STRATEGIES:

        print(f"\n====================")
        print(f"Strategy: {strategy_name}")
        print(f"====================")

        strategy = load_strategy(strategy_name)

        MODEL_PATH = ROOT / "models" / strategy.name / "final_model" / "model.pkl"
        FEATURES_PATH = ROOT / "models" / strategy.name / "final_model" / "features.pkl"

        DATA_PATH = ROOT / "data" / "processed" / f"{strategy.name}_{strategy.timeframe}_features.csv"

        # 2. BUILD FEATURES
        print("⚙️ Building features...")
        build_processed(strategy.name)

        # 3. LOAD DATA
        df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
        df = df.set_index(["Date", "Instrument"]).sort_index()

        # filter (defensive)
        df = df.loc[
            df.index.get_level_values("Instrument").isin(strategy.instruments)
        ]

        # latest date
        latest_date = df.index.get_level_values("Date").max()

        df_latest = df.loc[
            df.index.get_level_values("Date") == latest_date
        ]

        # 4. PREDICT
        predictor = Predictor(MODEL_PATH, FEATURES_PATH)
        predictions = predictor.predict(df_latest)

        # 5. SIGNALS
        signals = generate_signals(
            predictions,
            strategy.name,
            threshold=THRESHOLD,
            max_signals=MAX_SIGNALS
        )

        all_signals[strategy.name] = signals

        print(signals[["Instrument", "prob"]] if not signals.empty else "No signals")

    # =========================
    # EMAIL
    # =========================

    print("\n📧 Sending email...")

    message = format_message(all_signals)
    send_email(message)

    print("\n✅ DONE\n")


if __name__ == "__main__":
    main()