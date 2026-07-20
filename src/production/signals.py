def generate_signals(predictions, strategy_name, threshold=0.3, max_signals=8):

    df = predictions.copy()

    # reset index (Date, Instrument → columns)
    df = df.reset_index()

    #  dodaj strategiju
    df["strategy"] = strategy_name

    # FILTER
    df = df[df["prob"] >= threshold]

    if df.empty:
        return df

    # SORT
    df = df.sort_values("prob", ascending=False)

    # TOP N
    df = df.head(max_signals)

    return df