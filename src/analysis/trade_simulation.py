import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]


def simulate_trades(df, threshold=0.5, tp=0.03, sl=0.01, tp_atr=None, sl_atr=None, horizon=60):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values(["Instrument", "Date"])

    trades = []

    for instrument, group in df.groupby("Instrument"):
        group = group.reset_index(drop=True)

        in_position_until = -1

        for i in range(len(group) - 1):

            if i <= in_position_until:
                continue

            if group.loc[i, "y_proba"] < threshold:
                continue

            entry_idx = i + 1

            if entry_idx >= len(group):
                continue

            entry_date = group.loc[entry_idx, "Date"]
            entry_price = group.loc[entry_idx, "Open"]

            if tp_atr is not None and sl_atr is not None:

                atr = group.loc[entry_idx, "ATR_14"]

                tp_price = entry_price + tp_atr * atr
                sl_price = entry_price - sl_atr * atr

            else:

                tp_price = entry_price * (1 + tp)
                sl_price = entry_price * (1 - sl)

            max_exit_idx = min(entry_idx + horizon, len(group) - 1)

            exit_idx = None
            exit_price = None
            exit_reason = None

            for j in range(entry_idx, max_exit_idx + 1):
                high = group.loc[j, "High"]
                low = group.loc[j, "Low"]

                # konzervativno:
                # ako isti candle pogodi i TP i SL -> SL
                if low <= sl_price:
                    exit_idx = j
                    exit_price = sl_price
                    exit_reason = "SL"
                    break

                if high >= tp_price:
                    exit_idx = j
                    exit_price = tp_price
                    exit_reason = "TP"
                    break

            if exit_idx is None:
                exit_idx = max_exit_idx
                exit_price = group.loc[exit_idx, "Close"]
                exit_reason = "EXPIRED"

            exit_date = group.loc[exit_idx, "Date"]
            ret = (exit_price / entry_price) - 1

            trades.append({
                "Instrument": instrument,
                "Signal_Date": group.loc[i, "Date"],
                "Entry_Date": entry_date,
                "Exit_Date": exit_date,
                "Entry_Price": entry_price,
                "Exit_Price": exit_price,
                "Exit_Reason": exit_reason,
                "Prob": group.loc[i, "y_proba"],
                "Return": ret,
                "Holding_Bars": exit_idx - entry_idx + 1,
            })

            in_position_until = exit_idx

    return pd.DataFrame(trades)


def calculate_overlap(trades):
    if trades.empty:
        return 0, pd.DataFrame()

    events = []

    for _, row in trades.iterrows():
        events.append((row["Entry_Date"], 1))
        events.append((row["Exit_Date"], -1))

    events = sorted(events, key=lambda x: x[0])

    open_positions = 0
    rows = []

    for date, change in events:
        open_positions += change

        rows.append({
            "Date": date,
            "Open_Positions": open_positions
        })

    overlap_df = pd.DataFrame(rows)

    max_open = overlap_df["Open_Positions"].max()

    return max_open, overlap_df





def print_overlap_stats(overlap_df):
    if overlap_df.empty:
        return

    print("\n===== OPEN POSITION STATS =====")

    stats = overlap_df["Open_Positions"].describe()

    print(f"Average open positions: {stats['mean']:.2f}")
    print(f"Median open positions: {stats['50%']:.2f}")
    print(f"90th percentile: {overlap_df['Open_Positions'].quantile(0.9):.2f}")
    print(f"95th percentile: {overlap_df['Open_Positions'].quantile(0.95):.2f}")

    freq = (
        overlap_df["Open_Positions"]
        .value_counts(normalize=True)
        .sort_index()
        .head(15)
    )

    print("\nMost common open position counts:")

    for positions, pct in freq.items():
        print(f"{positions} positions -> {pct * 100:.2f}%")


def print_summary(trades, max_open_positions, overlap_df):
    if trades.empty:
        print("No trades generated.")
        return

    total = len(trades)

    win_rate = (trades["Return"] > 0).mean()

    avg_return = trades["Return"].mean()

    

    tp_count = (trades["Exit_Reason"] == "TP").sum()
    sl_count = (trades["Exit_Reason"] == "SL").sum()
    expired_count = (trades["Exit_Reason"] == "EXPIRED").sum()

    trades["Month"] = trades["Entry_Date"].dt.to_period("M")

    avg_trades_month = trades.groupby("Month").size().mean()

    

    print("\n===== TRADE SIMULATION SUMMARY =====")

    print(f"Trades: {total}")
    print(f"Win rate: {win_rate:.3f}")
    print(f"Avg return per trade: {avg_return:.4f}")
    
   

    # ovo JE po mjesecu i za intraday
    # samo računa broj tradeova unutar kalendarskog mjeseca
    print(f"Avg trades/month: {avg_trades_month:.2f}")

    print(f"Avg holding bars: {trades['Holding_Bars'].mean():.2f}")


    print(f"Max open positions: {max_open_positions}")

    print(f"TP hits: {tp_count}")
    print(f"SL hits: {sl_count}")
    print(f"Expired: {expired_count}")


    


    tp_trades = trades[
        trades["Exit_Reason"] == "TP"
    ]

    sl_trades = trades[
        trades["Exit_Reason"] == "SL"
    ]

    if len(tp_trades) > 0:

        print("\n===== TP DURATION STATS =====")

        avg_tp_duration = tp_trades['Holding_Bars'].mean()
        print(
            f"Avg TP duration: "
            f"{avg_tp_duration:.2f}"
        )

        print(
            f"Median TP duration: "
            f"{tp_trades['Holding_Bars'].median():.2f}"
        )

        print(
            f"90th percentile TP duration: "
            f"{tp_trades['Holding_Bars'].quantile(0.9):.2f}"
        )

    if len(sl_trades) > 0:

        print("\n===== SL DURATION STATS =====")

        avg_sl_duration = sl_trades['Holding_Bars'].mean()
        print(
            f"Avg SL duration: "
            f"{avg_sl_duration:.2f}"
        )

        print(
            f"Median SL duration: "
            f"{sl_trades['Holding_Bars'].median():.2f}"
        )

        print(
            f"90th percentile SL duration: "
            f"{sl_trades['Holding_Bars'].quantile(0.9):.2f}"
        )


    total_days = (
            trades["Entry_Date"].max()
            - trades["Entry_Date"].min()
    ).days

    signals_per_day = len(trades) / total_days

    expected_duration = (
            win_rate * avg_tp_duration
            + (1 - win_rate) * avg_sl_duration
    )

    expected_open_positions = (
            signals_per_day * expected_duration
    )

    avg_open_positions = overlap_df["Open_Positions"].mean()
    print("\n===== POSITION FLOW MODEL =====")

    print(f"Signals/day: {signals_per_day:.2f}")

    print(
        f"Expected duration: "
        f"{expected_duration:.2f}"
    )

    print(
        f"Expected open positions: "
        f"{expected_open_positions:.2f}"
    )

    print(
        f"Average open positions: "
        f"{avg_open_positions:.2f}"
    )
    print("\n===== PER INSTRUMENT =====")

    per_inst = trades.groupby("Instrument").agg(
        Trades=("Return", "count"),
        Win_Rate=("Return",
                lambda x: round((x > 0).mean(), 3)),
        Avg_Return=("Return",
                    lambda x: round(x.mean(), 4)),
        Median_Return=("Return",
                    lambda x: round(x.median(), 4)),
        Total_Return=("Return",
                    lambda x: round(x.sum(), 4)),
        Avg_Holding_Bars=("Holding_Bars",
                        lambda x: round(x.mean(), 2)),
    ).reset_index()

    print(per_inst.to_string(index=False))





def plot_open_positions(overlap_df, out_path):
    if overlap_df.empty:
        return

    counts = (
        overlap_df["Open_Positions"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(10, 5))

    plt.bar(
        counts.index.astype(str),
        counts.values
    )

    plt.title("Open Positions Distribution")
    plt.xlabel("Number of Open Positions")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(out_path)

    plt.close()

import json
import math


def save_strategy_profile(
    trades,
    overlap_df,
    output_path,
    threshold,
    horizon,
    tp,
    sl,
    tp_atr=None,
    sl_atr=None,
):
    if trades.empty:
        return

    wins = trades[trades["Return"] > 0]
    losses = trades[trades["Return"] <= 0]

    tp_trades = trades[trades["Exit_Reason"] == "TP"]
    sl_trades = trades[trades["Exit_Reason"] == "SL"]

    total_days = (
        trades["Entry_Date"].max()
        - trades["Entry_Date"].min()
    ).days

    total_days = max(total_days, 1)

    signals_per_day = len(trades) / total_days

    expected_duration = (
        trades["Holding_Bars"].mean()
    )

    avg_open_positions = (
        overlap_df["Open_Positions"].mean()
        if not overlap_df.empty
        else 0
    )

    expectancy = trades["Return"].mean()

    avg_win = (
        wins["Return"].mean()
        if len(wins) > 0
        else 0
    )

    avg_loss = (
        losses["Return"].mean()
        if len(losses) > 0
        else 0
    )

    payoff_ratio = (
        abs(avg_win / avg_loss)
        if avg_loss != 0
        else None
    )

    capital_turnover_rate = (
        1 / expected_duration
        if expected_duration > 0
        else None
    )

    capital_efficiency = (
        expectancy / expected_duration
        if expected_duration > 0
        else None
    )

    profile = {

        # =====================================================
        # STRATEGY INFO
        # =====================================================

        "strategy_info": {
            "threshold": threshold,
            "horizon": horizon,
            "tp": tp,
            "sl": sl,
            "tp_atr": tp_atr,
            "sl_atr": sl_atr,
        },

        # =====================================================
        # TRADE STATS
        # =====================================================

        "trade_statistics": {
            "n_trades": int(len(trades)),
            "win_rate": float(
                (trades["Return"] > 0).mean()
            ),
            "avg_return": float(
                trades["Return"].mean()
            ),
            "median_return": float(
                trades["Return"].median()
            ),
            "expectancy": float(expectancy),
        },

        # =====================================================
        # DURATION STATS
        # =====================================================

        "duration_statistics": {

            "avg_holding_bars": float(
                trades["Holding_Bars"].mean()
            ),

            "expected_duration": float(
                expected_duration
            ),

            # TP

            "avg_tp_duration": float(
                tp_trades["Holding_Bars"].mean()
            ) if len(tp_trades) else None,

            "median_tp_duration": float(
                tp_trades["Holding_Bars"].median()
            ) if len(tp_trades) else None,

            "p75_tp_duration": float(
                tp_trades["Holding_Bars"].quantile(0.75)
            ) if len(tp_trades) else None,

            "p90_tp_duration": float(
                tp_trades["Holding_Bars"].quantile(0.90)
            ) if len(tp_trades) else None,

            "p95_tp_duration": float(
                tp_trades["Holding_Bars"].quantile(0.95)
            ) if len(tp_trades) else None,

            # SL

            "avg_sl_duration": float(
                sl_trades["Holding_Bars"].mean()
            ) if len(sl_trades) else None,

            "median_sl_duration": float(
                sl_trades["Holding_Bars"].median()
            ) if len(sl_trades) else None,

            "p75_sl_duration": float(
                sl_trades["Holding_Bars"].quantile(0.75)
            ) if len(sl_trades) else None,

            "p90_sl_duration": float(
                sl_trades["Holding_Bars"].quantile(0.90)
            ) if len(sl_trades) else None,

            "p95_sl_duration": float(
                sl_trades["Holding_Bars"].quantile(0.95)
            ) if len(sl_trades) else None,
        },

        # =====================================================
        # OVERLAP STATS
        # =====================================================

        "overlap_statistics": {

            "avg_open_positions": float(
                avg_open_positions
            ),

            "median_open_positions": float(
                overlap_df["Open_Positions"].median()
            ) if not overlap_df.empty else 0,

            "p75_open_positions": float(
                overlap_df["Open_Positions"].quantile(0.75)
            ) if not overlap_df.empty else 0,

            "p90_open_positions": float(
                overlap_df["Open_Positions"].quantile(0.90)
            ) if not overlap_df.empty else 0,

            "p95_open_positions": float(
                overlap_df["Open_Positions"].quantile(0.95)
            ) if not overlap_df.empty else 0,

            "max_open_positions": int(
                overlap_df["Open_Positions"].max()
            ) if not overlap_df.empty else 0,
        },

        # =====================================================
        # SIGNAL STATS
        # =====================================================

        "signal_statistics": {

            "signals_per_day": float(
                signals_per_day
            ),

            "signals_per_month": float(
                signals_per_day * 30
            ),

            "avg_days_between_signals": float(
                total_days / len(trades)
            ),

            "signal_frequency_per_day": float(
                signals_per_day
            ),

            "signal_frequency_per_bar": float(
                len(trades) / total_days
            ),
        },

        # =====================================================
        # PAYOFF STATS
        # =====================================================

        "payoff_statistics": {

            "avg_win": float(avg_win),

            "avg_loss": float(avg_loss),

            "payoff_ratio": (
                float(payoff_ratio)
                if payoff_ratio is not None
                else None
            ),
        },

        # =====================================================
        # OUTCOME STATS
        # =====================================================

        "outcome_statistics": {

            "tp_hits": int(
                (trades["Exit_Reason"] == "TP").sum()
            ),

            "sl_hits": int(
                (trades["Exit_Reason"] == "SL").sum()
            ),

            "expired_hits": int(
                (trades["Exit_Reason"] == "EXPIRED").sum()
            ),

            "tp_probability": float(
                len(tp_trades) / len(trades)
            ),

            "sl_probability": float(
                len(sl_trades) / len(trades)
            ),

            "expired_probability": float(
                (trades["Exit_Reason"] == "EXPIRED").sum() / len(trades)
            ),
        },

        # =====================================================
        # RETURN DISTRIBUTION
        # =====================================================

        "distribution_statistics": {

            "return_std": float(
                trades["Return"].std()
            ),

            "return_q10": float(
                trades["Return"].quantile(0.10)
            ),

            "return_q25": float(
                trades["Return"].quantile(0.25)
            ),

            "return_q50": float(
                trades["Return"].quantile(0.50)
            ),

            "return_q75": float(
                trades["Return"].quantile(0.75)
            ),

            "return_q90": float(
                trades["Return"].quantile(0.90)
            ),
        },

        #capilat flow stat
        "capital_flow_statistics": {

            "expected_trade_duration": float(
                expected_duration
            ),

            "expected_tp_duration": float(
                tp_trades["Holding_Bars"].mean()
            ) if len(tp_trades) else None,

            "expected_sl_duration": float(
                sl_trades["Holding_Bars"].mean()
            ) if len(sl_trades) else None,

            "expected_capital_lock_days": float(
                expected_duration
            ),

            "capital_turnover_rate": (
                float(capital_turnover_rate)
                if capital_turnover_rate is not None
                else None
            ),

            "expected_open_positions": float(
                avg_open_positions
            ),
        },

        #risk stats
        "risk_statistics": {

            "return_std": float(
                trades["Return"].std()
            ),

            "return_variance": float(
                trades["Return"].var()
            ),

            "downside_std": float(
                losses["Return"].std()
            ) if len(losses) else None,

            "best_trade": float(
                trades["Return"].max()
            ),

            "worst_trade": float(
                trades["Return"].min()
            ),
        },

        #holding dist
        "holding_distribution": {

            "tp_duration_histogram": tp_trades["Holding_Bars"].value_counts().sort_index().to_dict(),

            "sl_duration_histogram": sl_trades["Holding_Bars"].value_counts().sort_index().to_dict(),
        },

        # =====================================================
        # ALLOCATOR INPUTS
        # =====================================================

        "allocator_inputs": {

            "expected_open_positions": float(
                avg_open_positions
            ),

            "capital_turnover_rate": (
                float(capital_turnover_rate)
                if capital_turnover_rate is not None
                else None
            ),

            "capital_efficiency": (
                float(capital_efficiency)
                if capital_efficiency is not None
                else None
            ),
        },

        # =====================================================
        # FUTURE FEATURES
        # =====================================================

        "future_features": {

            "instrument_count": int(
                trades["Instrument"].nunique()
            ),

            "best_instrument": (
                trades.groupby("Instrument")["Return"]
                .mean()
                .idxmax()
            ),

            "worst_instrument": (
                trades.groupby("Instrument")["Return"]
                .mean()
                .idxmin()
            ),
        }
    }

    with open(output_path, "w") as f:
        json.dump(
            profile,
            f,
            indent=4
        )

    print(
        f"\nStrategy profile saved to: {output_path}"
    )

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--predictions", required=True)

    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--tp", type=float, default=0.03)

    parser.add_argument("--sl", type=float, default=0.01)

    parser.add_argument("--tp-atr", type=float, default=None)

    parser.add_argument("--sl-atr", type=float,default=None)

    parser.add_argument("--horizon", type=int, default=60)

    

    

    args = parser.parse_args()

    pred_path = ROOT / args.predictions

    df = pd.read_csv(
        pred_path,
        parse_dates=["Date"]
    )

    trades = simulate_trades(
        df,
        threshold=args.threshold,
        tp=args.tp,
        sl=args.sl,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
        horizon=args.horizon
    )

    

    max_open_positions, overlap = calculate_overlap(trades)

    out_dir = pred_path.parent

    trades_path = out_dir / "trades_simulation.csv"


    overlap_path = out_dir / "open_positions.csv"


    overlap_plot_path = out_dir / "open_positions_distribution.png"

    trades.to_csv(trades_path, index=False)

    

    overlap.to_csv(overlap_path, index=False)

    profile_path = out_dir / "strategy_profile.json"
    

    save_strategy_profile(
        trades=trades,
        overlap_df=overlap,
        output_path=profile_path,
        threshold=args.threshold,
        horizon=args.horizon,
        tp=args.tp,
        sl=args.sl,
        tp_atr=args.tp_atr,
        sl_atr=args.sl_atr,
)

    

    plot_open_positions(overlap, overlap_plot_path)

    print_summary(
        trades,
        max_open_positions,
        overlap
    )

    print_overlap_stats(overlap)

    print(f"\nTrades saved to: {trades_path}")

    

    print(f"Open positions saved to: {overlap_path}")

    

    print(f"Open positions distribution plot saved to: {overlap_plot_path}")

    print(f"Strategy profile saved to: "f"{profile_path}")

if __name__ == "__main__":
    main()