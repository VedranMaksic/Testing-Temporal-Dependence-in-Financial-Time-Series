import pandas as pd
import matplotlib
matplotlib.use("Agg")  # brže renderiranje (bez GUI)

import matplotlib.pyplot as plt
from pathlib import Path
import argparse


ROOT = Path(__file__).resolve().parents[2]


def plot_instrument(file_path: Path, output_dir: Path):

    df = pd.read_csv(file_path, parse_dates=["Date"])
    df = df.sort_values("Date")

    name = file_path.stem

    # ubrzanje (downsample)
    df = df.iloc[::5]

    # volume manji (height ratio)
    fig, (ax1, ax2) = plt.subplots(
        2, 1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={'height_ratios': [3, 1]}
    )

    # ======================
    # PRICE
    # ======================
    ax1.plot(df["Date"], df["Close"], linewidth=1)
    ax1.set_title(f"{name} - Price")
    ax1.set_ylabel("Price")

    # ======================
    # VOLUME
    # ======================
    ax2.plot(df["Date"], df["Volume"], alpha=0.5)
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")

    # ======================
    # SAVE
    # ======================
    out_path = output_dir / f"{name}.png"

    plt.subplots_adjust(hspace=0.2)
    plt.savefig(out_path)
    plt.close()

    print(f"Saved: {out_path}")


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--timeframe", default="1d", help="npr. 1d, 1h, 30m, 15m")
    args = parser.parse_args()

    timeframe = args.timeframe

    DATA_RAW = ROOT / "data" / "raw" / timeframe
    OUTPUT_DIR = ROOT / "reports" / "plots" / timeframe

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    files = list(DATA_RAW.glob("*.csv"))

    if not files:
        print("No data found")
        return

    for file in files:
        plot_instrument(file, OUTPUT_DIR)


if __name__ == "__main__":
    main()