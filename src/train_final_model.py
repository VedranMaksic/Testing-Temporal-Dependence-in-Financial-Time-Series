import argparse
import importlib
import pandas as pd
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


# =========================
# LOAD STRATEGY
# =========================

def load_strategy(strategy_name):
    module = importlib.import_module(f"src.strategies.{strategy_name}")
    return module.Strategy()


def load_target_module(strategy):
    return importlib.import_module(f"src.targets.{strategy.target['module']}")


def load_model_module(strategy):
    return importlib.import_module(f"src.trainers.{strategy.model['module']}")


# =========================
# MAIN
# =========================

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy", required=True)
    args = parser.parse_args()

    strategy = load_strategy(args.strategy)

    print("\n🚀 TRAIN FINAL MODEL")
    print(f"Strategy: {strategy.name}")
    print(f"Instruments: {strategy.instruments}")

    # =========================
    # LOAD FEATURES
    # =========================

    data_path = ROOT / "data" / "processed" / f"{strategy.name}_{strategy.timeframe}_features.csv"

    df = pd.read_csv(data_path, parse_dates=["Date"])
    df = df.set_index(["Date", "Instrument"]).sort_index()

    print(f"Loaded data: {df.shape}")

    # =========================
    # FILTER INSTRUMENTS
    # =========================

    if hasattr(strategy, "instruments") and strategy.instruments:
        df = df.loc[
            df.index.get_level_values("Instrument").isin(strategy.instruments)
        ]

    print(f"Filtered data: {df.shape}")

    # =========================
    # TARGET
    # =========================

    target_module = load_target_module(strategy)

    df = target_module.generate(
        df,
        **strategy.target["params"]
    )

    df = df.dropna(subset=["target"])
    df = df.sort_index()

    # =========================
    # FEATURES
    # =========================

    feature_cols = list(strategy.features.keys())

    X = df[feature_cols]
    y = df["target"]

    print(f"Training rows: {len(X)}")

    # =========================
    # TRAIN MODEL
    # =========================

    model_module = load_model_module(strategy)
    trainer = model_module.Trainer(**strategy.model["params"])

    print("🤖 Training model...")
    trainer.train(X, y)

    # =========================
    # SAVE MODEL
    # =========================

    out_dir = ROOT / "models" / strategy.name / "final_model"
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = out_dir / "model.pkl"
    features_path = out_dir / "features.pkl"

    joblib.dump(trainer.model, model_path)
    joblib.dump(feature_cols, features_path)

    print(f"\n✅ Model saved: {model_path}")
    print(f"✅ Features saved: {features_path}")


if __name__ == "__main__":
    main()