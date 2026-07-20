import argparse
import json
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load_allocation_profile(path):
    with open(path, "r") as f:
        return json.load(f)


def build_portfolio_simulation(
    trades,
    allocation_profile,
    initial_capital=10000,
):
    trades = trades.copy()

    if trades.empty:
        return pd.DataFrame(), pd.DataFrame(), {}

    trades["Entry_Date"] = pd.to_datetime(
        trades["Entry_Date"]
    )

    trades["Exit_Date"] = pd.to_datetime(
        trades["Exit_Date"]
    )

    trades = trades.sort_values(
        "Entry_Date"
    ).reset_index(drop=True)

    slots = allocation_profile[
        "recommended_live_params"
    ]["slots_to_use"]

    usage_target = allocation_profile[
        "capital_policy"
    ]["capital_usage_target"]

    cash = initial_capital

    open_positions = []

    trade_rows = []
    equity_rows = []

    skipped_trades = 0

    for _, trade in trades.iterrows():

        current_date = trade["Entry_Date"]

        # ---------------------------------
        # CLOSE POSITIONS
        # ---------------------------------

        still_open = []

        for pos in open_positions:

            if pos["Exit_Date"] <= current_date:

                cash += (
                    pos["Trade_Size"]
                    + pos["Profit"]
                )

            else:

                still_open.append(pos)

        open_positions = still_open

        # ---------------------------------
        # CURRENT EQUITY
        # ---------------------------------

        locked_capital = sum(
            p["Trade_Size"]
            for p in open_positions
        )

        equity = cash + locked_capital

        # ---------------------------------
        # REBALANCING SLOT SIZE
        # ---------------------------------

        trade_size = (
            equity
            * usage_target
            / slots
        )

        # ---------------------------------
        # CAPITAL CHECK
        # ---------------------------------

        if cash < trade_size:

            skipped_trades += 1

            trade_rows.append({
                "Instrument": trade["Instrument"],
                "Entry_Date": trade["Entry_Date"],
                "Exit_Date": trade["Exit_Date"],
                "Return": trade["Return"],
                "Trade_Size": 0,
                "Profit": 0,
                "Skipped": True,
                "Cash_Before": cash,
                "Cash_After": cash,
            })

            continue

        # ---------------------------------
        # OPEN POSITION
        # ---------------------------------

        cash_before = cash

        cash -= trade_size

        profit = (
            trade_size
            * trade["Return"]
        )

        open_positions.append({
            "Exit_Date": trade["Exit_Date"],
            "Trade_Size": trade_size,
            "Profit": profit,
        })

        trade_rows.append({
            "Instrument": trade["Instrument"],
            "Entry_Date": trade["Entry_Date"],
            "Exit_Date": trade["Exit_Date"],
            "Return": trade["Return"],
            "Trade_Size": trade_size,
            "Profit": profit,
            "Skipped": False,
            "Cash_Before": cash_before,
            "Cash_After": cash,
        })

        locked_capital = sum(
            p["Trade_Size"]
            for p in open_positions
        )

        equity_rows.append({
            "Date": current_date,
            "Cash": cash,
            "Locked_Capital": locked_capital,
            "Equity": cash + locked_capital,
            "Open_Positions": len(open_positions),
        })

    # ---------------------------------
    # FINAL SETTLEMENT
    # ---------------------------------

    for pos in open_positions:

        cash += (
            pos["Trade_Size"]
            + pos["Profit"]
        )

    trades_df = pd.DataFrame(
        trade_rows
    )

    equity_df = pd.DataFrame(
        equity_rows
    )

    if not equity_df.empty:

        equity_df["Peak"] = (
            equity_df["Equity"]
            .cummax()
        )

        equity_df["Drawdown"] = (
            equity_df["Equity"]
            / equity_df["Peak"]
        ) - 1

    final_equity = cash

    total_profit = (
        final_equity
        - initial_capital
    )

    summary = {

        "initial_capital":
            float(initial_capital),

        "final_equity":
            float(final_equity),

        "total_profit":
            float(total_profit),

        "total_trades":
            int(len(trades_df)),

        "executed_trades":
            int(
                (~trades_df["Skipped"]).sum()
            ),

        "skipped_trades":
            int(skipped_trades),

        "avg_trade_size":
            float(
                trades_df.loc[
                    ~trades_df["Skipped"],
                    "Trade_Size"
                ].mean()
            ),

        "max_drawdown":
            float(
                equity_df["Drawdown"].min()
            )
            if not equity_df.empty
            else 0,

        "max_open_positions":
            int(
                equity_df["Open_Positions"].max()
            )
            if not equity_df.empty
            else 0,
    }

    return (
        trades_df,
        equity_df,
        summary
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True
    )

    parser.add_argument(
        "--initial-capital",
        type=float,
        default=10000
    )

    args = parser.parse_args()

    model_dir = ROOT / args.model

    trades_path = (
        model_dir
        / "trades_simulation.csv"
    )

    allocation_path = (
        model_dir
        / "allocation_profile.json"
    )

    trades = pd.read_csv(
        trades_path,
        parse_dates=[
            "Entry_Date",
            "Exit_Date"
        ]
    )

    allocation = (
        load_allocation_profile(
            allocation_path
        )
    )

    (
        portfolio_trades,
        equity_curve,
        summary
    ) = build_portfolio_simulation(
        trades,
        allocation,
        args.initial_capital
    )

    portfolio_trades.to_csv(
        model_dir / "portfolio_trades.csv",
        index=False
    )

    equity_curve.to_csv(
        model_dir / "equity_curve.csv",
        index=False
    )

    with open(
        model_dir
        / "portfolio_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=4
        )

    print(
        json.dumps(
            summary,
            indent=4
        )
    )


if __name__ == "__main__":
    main()