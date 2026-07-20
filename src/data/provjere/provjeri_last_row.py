import pandas as pd

CSV_PATH = "data/processed/all_instruments_features_enhanced.csv"
INSTRUMENT = "CROBEX"   

df = pd.read_csv(CSV_PATH, parse_dates=["Date"])

last_row = (
    df[df["Instrument"] == INSTRUMENT]
    .sort_values("Date")
    .tail(1)
)

print(f"\nLast row for {INSTRUMENT}:")
print(last_row[["Date", "Instrument"]])

print("\nNaN count per column:")
print(last_row.isna().sum()[last_row.isna().sum() > 0])

print("\nColumns with NaN:")
print(last_row.columns[last_row.isna().any()])
