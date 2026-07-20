import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/up5_daily_enhanced/predictions.csv")
df["Date"] = pd.to_datetime(df["Date"])

btc = df[df["Instrument"] == "Bitcoin"].copy()

btc["Year"] = btc["Date"].dt.year
print(btc.groupby("Year")["target"].sum())

plt.figure(figsize=(12,5))
plt.plot(btc["Date"], btc["y_proba"])
plt.title("Bitcoin – Predicted Probability Over Time")
plt.axhline(0.3, color="red", linestyle="--")
plt.show()