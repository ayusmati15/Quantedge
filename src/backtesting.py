import numpy as np
import pandas as pd
from config import *
from performance_metrics import calculate_metrics

# Apply transaction costs
def apply_transaction_costs(df):

    df["Trade"] = df["Position"].diff().abs().fillna(0)

    df["Transaction_Cost"] = df["Trade"] * TRANSACTION_COST

    return df

# Apply slippage
def apply_slippage(df):

    df["Slippage"] = df["Trade"] * SLIPPAGE

    return df

# Backtest strategy
def backtest(df):

    df = df.copy()

    df["Market_Return"] = df["Close"].pct_change().fillna(0)

    df["Strategy_Return"] = (
        df["Position"].shift(1).fillna(0)
        * df["Market_Return"]
    )

    df = apply_transaction_costs(df)

    df = apply_slippage(df)

    df["Strategy_Return"] -= (
        df["Transaction_Cost"] +
        df["Slippage"]
    )

    df["Cumulative_Market"] = (
        1 + df["Market_Return"]
    ).cumprod()

    df["Cumulative_Strategy"] = (
        1 + df["Strategy_Return"]
    ).cumprod()

    return df

# Trade statistics
def trade_statistics(df):

    wins = (df["Strategy_Return"] > 0).sum()

    losses = (df["Strategy_Return"] < 0).sum()

    total = wins + losses

    return {
        "Trades": int(total),
        "Wins": int(wins),
        "Losses": int(losses),
        "Win Rate": wins / total if total else 0
    }

# Monthly returns
def monthly_returns(df):

    monthly = (
        df["Strategy_Return"]
        .resample("M")
        .sum()
    )

    return monthly

# Complete pipeline
def run_backtest(df, benchmark=None):

    df = backtest(df)

    metrics = calculate_metrics(
        df["Cumulative_Strategy"].values,
        benchmark
    )

    stats = trade_statistics(df)

    return {
        "data": df,
        "metrics": metrics,
        "statistics": stats
    }

if __name__ == "__main__":
    print("Backtesting Module Loaded")
