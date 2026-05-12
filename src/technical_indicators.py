import pandas as pd


def add_rsi(df, window=14):

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)

    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(window=window).mean()

    avg_loss = loss.rolling(window=window).mean()

    rs = avg_gain / avg_loss

    df["RSI"] = 100 - (
        100 / (1 + rs)
    )

    return df


def add_macd(df):

    ema12 = df["Close"].ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = df["Close"].ewm(
        span=26,
        adjust=False
    ).mean()

    df["MACD"] = ema12 - ema26

    df["MACD_Signal"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    return df


def add_bollinger_bands(
    df,
    window=20
):

    rolling_mean = (
        df["Close"]
        .rolling(window)
        .mean()
    )

    rolling_std = (
        df["Close"]
        .rolling(window)
        .std()
    )

    df["BB_Upper"] = (
        rolling_mean
        + 2 * rolling_std
    )

    df["BB_Lower"] = (
        rolling_mean
        - 2 * rolling_std
    )

    return df