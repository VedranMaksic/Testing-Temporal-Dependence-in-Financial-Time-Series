# src/production/test_predictor.py
from signals import generate_signals
import pandas as pd
from pathlib import Path

from predictor import Predictor

ROOT = Path(__file__).resolve().parents[2]

# =========================
# PATHS (prilagodi strategy)
# =========================

MODEL_PATH = ROOT / "models" / "up5_daily_enhanced" / "final_model" / "model.pkl"
FEATURES_PATH = ROOT / "models" / "up5_daily_enhanced" / "final_model" / "features.pkl"

DATA_PATH = ROOT / "data" / "processed" / "up5_daily_enhanced_1d_features.csv"

def main():

    print("📥 Loading data...")

    df = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    df = df.set_index(["Date", "Instrument"]).sort_index()

    print("Data loaded:", df.shape)

    print("🤖 Loading predictor...")

    predictor = Predictor(MODEL_PATH, FEATURES_PATH)

    print("🔮 Running prediction...")

    predictions = predictor.predict(df)

    print("\n✅ Predictions:")

    print(predictions[["prob"]].head(10))

    signals = generate_signals(predictions, threshold=0.3)

    print("\n🚀 SIGNALS:")
    print(signals)  
if __name__ == "__main__":
    main()