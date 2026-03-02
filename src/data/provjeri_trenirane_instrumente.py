import pandas as pd
from src.models.config import Config

# --------------------------------------------------
# Učitaj config (isti koji koristi trening)
# --------------------------------------------------
cfg = Config()

print("Using CSV:", cfg.input_csv)
print("Train period:", cfg.train_start, "→", cfg.train_end)
print("Test start:", cfg.test_start)
print("-" * 60)


df = pd.read_csv(cfg.input_csv, parse_dates=["Date"])


df = df.set_index("Date").sort_index()

print("Full dataset date range:")
print(df.index.min(), "→", df.index.max())
print("Total rows:", len(df))
print("-" * 60)


train_df = df.loc[cfg.train_start : cfg.train_end]
test_df = df.loc[cfg.test_start :]


print("=== TRAIN SET ===")
print("Rows total:", len(train_df))
print("Rows per instrument:")
print(train_df["Instrument"].value_counts())

print("\n=== TEST SET ===")
print("Rows total:", len(test_df))
print("Rows per instrument:")
print(test_df["Instrument"].value_counts())

# --------------------------------------------------
# Posebna provjera za Bitcoin
# --------------------------------------------------
btc_train = train_df[train_df["Instrument"] == "Bitcoin"]
btc_test = test_df[test_df["Instrument"] == "Bitcoin"]

print("\n=== BITCOIN DETAIL ===")
print("Bitcoin rows in TRAIN:", len(btc_train))
if len(btc_train) > 0:
    print("Bitcoin TRAIN date range:",
          btc_train.index.min(), "→", btc_train.index.max())
else:
    print("Bitcoin NOT present in TRAIN period")

print("\nBitcoin rows in TEST:", len(btc_test))
if len(btc_test) > 0:
    print("Bitcoin TEST date range:",
          btc_test.index.min(), "→", btc_test.index.max())
else:
    print("Bitcoin NOT present in TEST period")
