import numpy as np


def generate_signals(df):

    df = df.copy()

    df["Signal"] = (
        df["MA_10"] > df["MA_20"]
    ).astype(int)

    df["Position"] = df["Signal"].shift(1)

    df["Position"] = df["Position"].fillna(0)

    return df


def backtest_strategy(df):

    df = generate_signals(df)

    df["Market_Return"] = df["Close"].pct_change().fillna(0)

    df["Strategy_Return"] = (
        df["Market_Return"] * df["Position"]
    )

    df["Cumulative_Market"] = (
        (1 + df["Market_Return"]).cumprod()
    )

    df["Cumulative_Strategy"] = (
        (1 + df["Strategy_Return"]).cumprod()
    )

    return df


def performance_metrics(df):

    strategy_return = (
        df["Cumulative_Strategy"].iloc[-1] - 1
    )

    market_return = (
        df["Cumulative_Market"].iloc[-1] - 1
    )

    sharpe = (
        df["Strategy_Return"].mean()
        / (df["Strategy_Return"].std() + 1e-9)
    ) * np.sqrt(252)

    max_drawdown = (
        df["Cumulative_Strategy"]
        / df["Cumulative_Strategy"].cummax()
        - 1
    ).min()

    return {
        "Strategy Return": strategy_return,
        "Market Return": market_return,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_drawdown
    }