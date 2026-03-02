import json
import pandas as pd
import numpy as np

# --------------------------------------------------
# KONFIGURACIJA – koristi ISTO kao predict_model
# --------------------------------------------------

VARIANT = "enhanced"  # "baseline" ili "enhanced"

BASE_CSV = "data/processed/all_instruments_features_base.csv"
ENH_CSV  = "data/processed/all_instruments_features_enhanced.csv"

MODEL_BASE = "models/final_baseline"
MODEL_ENH  = "models/final_enhanced"

if VARIANT == "baseline":
    csv_path = BASE_CSV
    model_dir = MODEL_BASE
else:
    csv_path = ENH_CSV
    model_dir = MODEL_ENH


print("=" * 80)
print("DEBUGGING CROBEX PREDICTION")
print("Variant:", VARIANT)
print("CSV:", csv_path)
print("Model dir:", model_dir)
print("=" * 80)

# ucitaj

df = pd.read_csv(csv_path, parse_dates=["Date"])
df = df.set_index("Date").sort_index()

print("\n[1] INSTRUMENTI U CSV-U:")
print(sorted(df["Instrument"].unique()))

if "CROBEX" not in df["Instrument"].unique():
    print("❌ CROBEX NIJE U CSV-U → PROBLEM JE U ETL-u")
    raise SystemExit

print("✅ CROBEX JE U CSV-U")

# ucitaj feature

with open(f"{model_dir}/feature_columns.json", "r", encoding="utf-8") as f:
    feature_cols = json.load(f)

print("\n[2] BROJ FEATUREA U MODELU:", len(feature_cols))

# nadi crobex

df_crobex = df[df["Instrument"] == "CROBEX"].copy()

print("\n[3] CROBEX TOTAL ROWS:", len(df_crobex))
print("Date range:", df_crobex.index.min(), "→", df_crobex.index.max())

#nap sto bi i predikcija

tmp = df_crobex.copy()

inst_oh = pd.get_dummies(tmp["Instrument"], prefix="Inst", dtype=int)
tmp = pd.concat([tmp.drop(columns=["Instrument"]), inst_oh], axis=1)
tmp = tmp.select_dtypes(include=[np.number])
tmp = tmp.replace([np.inf, -np.inf], np.nan)

# Dodaj feature kolone koje model očekuje, a ne postoje
for c in feature_cols:
    if c not in tmp.columns:
        tmp[c] = 0

tmp = tmp[feature_cols]

nan_rows = tmp.isna().any(axis=1)

print("\n[4] NaN ANALIZA ZA CROBEX:")
print("Rows with ANY NaN:", nan_rows.sum())
print("Rows without NaN:", (~nan_rows).sum())

if (~nan_rows).sum() == 0:
    print("❌ NIJEDAN VALJAN RED ZA CROBEX → FEATURE BUG")
    print("\nFeaturei s NaN vrijednostima (zadnjih 10 redova):")
    print(tmp.tail(10).isna().sum())
    raise SystemExit

# ZADNJI VALJAN RED 

last_valid_date = tmp.loc[~nan_rows].index.max()

print("\n[5] ZADNJI VALJAN DATUM ZA CROBEX:", last_valid_date)

print("\nFeature NaN status u tom retku:")
print(tmp.loc[last_valid_date].isna().sum(), "NaN vrijednosti")

#provjera

with open(f"{model_dir}/meta.json", "r", encoding="utf-8") as f:
    meta = json.load(f)

print("\n[6] FINAL MODEL META:")
for k, v in meta.items():
    print(f"{k}: {v}")

print("\n✅ AKO SI DOŠAO DO OVOGA, CROBEX IMA VALJAN RED ZA PREDIKCIJU")
print("Ako ga nema u predict outputu → bug je 100% u predict_model.py")
