import pandas as pd

CSV_PATH = "data/processed/all_instruments_features_enhanced.csv"

df = pd.read_csv(CSV_PATH)

print("=== Unique instruments in PROCESSED CSV ===")
print(sorted(df["Instrument"].dropna().unique()))

print("\nRows per instrument:")
print(df["Instrument"].value_counts())
