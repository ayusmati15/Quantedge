import pandas as pd


def create_features(df):

    df = df.copy()

    df["Returns"] = df["Close"].pct_change()

    df["MA_10"] = (
        df["Close"]
        .rolling(window=10)
        .mean()
    )

    df["MA_20"] = (
        df["Close"]
        .rolling(window=20)
        .mean()
    )

    df["Volatility"] = (
        df["Returns"]
        .rolling(window=10)
        .std()
    )

    df["Momentum"] = (
        df["Close"]
        - df["Close"].shift(5)
    )

    df["Lag_1"] = (
        df["Close"].shift(1)
    )

    df["Lag_2"] = (
        df["Close"].shift(2)
    )

    df["Target"] = (
        df["Close"].shift(-1)
        > df["Close"]
    ).astype(int)

    df.dropna(inplace=True)

    return df