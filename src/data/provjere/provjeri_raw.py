import pandas as pd

CSV_PATH = "data/raw/all_instruments_raw.csv"

df = pd.read_csv(CSV_PATH)

print("=== Unique instruments in RAW CSV ===")
print(sorted(df["Instrument"].dropna().unique()))

print("\nRows per instrument:")
print(df["Instrument"].value_counts())
