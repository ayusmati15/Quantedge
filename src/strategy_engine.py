import numpy as np


def rsi_strategy(df):

    df = df.copy()

    df["RSI_Signal"] = 0

    df.loc[
        df["RSI"] < 30,
        "RSI_Signal"
    ] = 1

    df.loc[
        df["RSI"] > 70,
        "RSI_Signal"
    ] = -1

    return df


def macd_strategy(df):

    df = df.copy()

    df["MACD_Signal_Strategy"] = 0

    df.loc[
        df["MACD"] > df["MACD_Signal"],
        "MACD_Signal_Strategy"
    ] = 1

    df.loc[
        df["MACD"] < df["MACD_Signal"],
        "MACD_Signal_Strategy"
    ] = -1

    return df


def hybrid_strategy(df):

    df = rsi_strategy(df)

    df = macd_strategy(df)

    df["Hybrid_Signal"] = 0

    buy_condition = (
        (df["RSI_Signal"] == 1)
        &
        (df["MACD_Signal_Strategy"] == 1)
    )

    sell_condition = (
        (df["RSI_Signal"] == -1)
        &
        (df["MACD_Signal_Strategy"] == -1)
    )

    df.loc[
        buy_condition,
        "Hybrid_Signal"
    ] = 1

    df.loc[
        sell_condition,
        "Hybrid_Signal"
    ] = -1

    return df


def calculate_strategy_returns(df):

    df = df.copy()

    df["Market_Return"] = (
        df["Close"]
        .pct_change()
        .fillna(0)
    )

    df["Strategy_Return"] = (
        df["Hybrid_Signal"]
        .shift(1)
        .fillna(0)
        *
        df["Market_Return"]
    )

    df["Cumulative_Strategy"] = (
        1 + df["Strategy_Return"]
    ).cumprod()

    return df