import numpy as np
import pandas as pd
from config import *

# Daily returns
def daily_returns(prices):
    return pd.Series(prices).pct_change().dropna()

# Cumulative returns
def cumulative_returns(returns):
    return (1 + returns).cumprod()

# CAGR
def cagr(prices):

    years = len(prices) / 252

    return (prices[-1] / prices[0]) ** (1 / years) - 1

# Annual volatility
def volatility(returns):
    return returns.std() * np.sqrt(252)

# Sharpe Ratio
def sharpe_ratio(returns):

    excess = returns - (RISK_FREE_RATE / 252)

    return np.sqrt(252) * excess.mean() / excess.std()

# Sortino Ratio
def sortino_ratio(returns):

    downside = returns[returns < 0]

    return np.sqrt(252) * returns.mean() / downside.std()

# Maximum Drawdown
def max_drawdown(prices):

    prices = np.array(prices)

    peak = np.maximum.accumulate(prices)

    drawdown = (prices - peak) / peak

    return drawdown.min()

# Calmar Ratio
def calmar_ratio(prices):

    return cagr(prices) / abs(max_drawdown(prices))

# Win Rate
def win_rate(returns):

    return (returns > 0).sum() / len(returns)

# Profit Factor
def profit_factor(returns):

    profit = returns[returns > 0].sum()

    loss = abs(returns[returns < 0].sum())

    return np.inf if loss == 0 else profit / loss

# Alpha & Beta
def alpha_beta(strategy, benchmark):

    beta = np.cov(strategy, benchmark)[0,1] / np.var(benchmark)

    alpha = strategy.mean() - beta * benchmark.mean()

    return alpha, beta

# Information Ratio
def information_ratio(strategy, benchmark):

    diff = strategy - benchmark

    return diff.mean() / diff.std()

# Metrics dictionary
def calculate_metrics(prices, benchmark=None):

    returns = daily_returns(prices)

    metrics = {

        "CAGR": cagr(prices),

        "Volatility": volatility(returns),

        "Sharpe": sharpe_ratio(returns),

        "Sortino": sortino_ratio(returns),

        "Calmar": calmar_ratio(prices),

        "Max Drawdown": max_drawdown(prices),

        "Win Rate": win_rate(returns),

        "Profit Factor": profit_factor(returns)

    }

    if benchmark is not None:

        benchmark_returns = daily_returns(benchmark)

        alpha, beta = alpha_beta(

            returns.values,

            benchmark_returns.values[:len(returns)]

        )

        metrics["Alpha"] = alpha

        metrics["Beta"] = beta

        metrics["Information Ratio"] = information_ratio(

            returns.values,

            benchmark_returns.values[:len(returns)]

        )

    return metrics

# Print metrics
def print_metrics(metrics):

    print("\nPerformance Metrics")

    print("-" * 30)

    for k, v in metrics.items():

        print(f"{k:<20}: {v:.4f}")

# Export
def export_metrics(metrics):

    df = pd.DataFrame([metrics])

    path = REPORT_DIRECTORY + "/performance_metrics.csv"

    df.to_csv(path, index=False)

    print("Saved:", path)

# Pipeline
def run_performance_analysis(prices, benchmark=None):

    metrics = calculate_metrics(prices, benchmark)

    print_metrics(metrics)

    export_metrics(metrics)

    return metrics


if __name__ == "__main__":

    print("Performance Metrics Loaded")
