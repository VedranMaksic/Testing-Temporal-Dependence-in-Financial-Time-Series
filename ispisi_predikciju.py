import pandas as pd
from pathlib import Path

MODELS_DIR = Path(
    r"C:\Users\maksi\Desktop\fer\3. godina apsolvent\zavrsni rad\project\models"
)


files = list(MODELS_DIR.glob("predictions_live_enhanced_*.csv"))
if not files:
    raise FileNotFoundError("Nema predictions_live_enhanced CSV datoteka.")

latest_file = max(files, key=lambda f: f.stat().st_mtime)

print("=" * 120)
print(f"📄 Najnoviji prediction file: {latest_file.name}")
print("=" * 120)

df = pd.read_csv(latest_file)


df["MaxPrice60_pred"] = df["Close"] * (1 + df["MaxRet60_pred"])
df["MinPrice60_pred"] = df["Close"] * (1 + df["MinRet60_pred"])


pd.set_option("display.max_columns", None)
pd.set_option("display.width", 220)
pd.set_option("display.float_format", "{:.4f}".format)


cols = [
    "Date",
    "Instrument",
    "Close",
    "P_Up10",
    "P_Down10",
    "MaxRet60_pred",
    "MaxPrice60_pred",
    "MinRet60_pred",
    "MinPrice60_pred"
]

print(df[cols].to_string(index=False))

print("=" * 120)
print(f"Ukupno redaka: {len(df)}")
