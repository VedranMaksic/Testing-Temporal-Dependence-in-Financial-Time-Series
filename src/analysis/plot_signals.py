import argparse
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--show_volume", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.predictions, parse_dates=["Date"])

    df = df[df["Instrument"] == args.instrument].copy()
    df = df.sort_values("Date")

    if df.empty:
        print("No data for this instrument.")
        return

    # === TRUE POSITIVE / FALSE POSITIVE ===
    df["TP"] = (df["y_pred_0_5"] == 1) & (df["y_true"] == 1)
    df["FP"] = (df["y_pred_0_5"] == 1) & (df["y_true"] == 0)

    fig, axes = plt.subplots(
        2 if args.show_volume else 1,
        1,
        figsize=(14, 8),
        sharex=True
    )

    if not args.show_volume:
        axes = [axes]

    # ===== PRICE =====
    axes[0].plot(df["Date"], df["Close"], linewidth=1, label="Close")

    axes[0].scatter(
        df[df["TP"]]["Date"],
        df[df["TP"]]["Close"],
        color="green",
        label="True Positive",
        s=40
    )

    axes[0].scatter(
        df[df["FP"]]["Date"],
        df[df["FP"]]["Close"],
        color="red",
        label="False Positive",
        s=40
    )

    axes[0].set_title(f"{args.instrument} - Signals")
    axes[0].legend()
    axes[0].grid(True)

    # ===== VOLUME =====
    if args.show_volume and "Volume" in df.columns:
        axes[1].bar(df["Date"], df["Volume"])
        axes[1].set_title("Volume")
        axes[1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()