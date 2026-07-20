import pandas as pd


def generate(df: pd.DataFrame, direction: str, horizon: int, threshold: float):

    df = df.copy()

    future_return = (
        df.groupby(level="Instrument")["Close"]
        .shift(-horizon) / df["Close"] - 1
    )

    if direction == "up":
        df["target"] = (future_return >= threshold).astype(int)
    else:
        df["target"] = (future_return <= -threshold).astype(int)

    return df