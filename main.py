import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd

from config import *

from feature_engineering import create_features
from technical_indicators import (
    add_rsi,
    add_macd,
    add_bollinger_bands
)

from regime_detection import run_regime_detection
from models import run_models
from signal_generator import run_signal_generator
from strategy_engine import run_strategy
from portfolio import optimize_portfolio
from backtesting import run_backtest
from performance_metrics import print_metrics
from visualization import (
    plot_equity_curve,
    plot_drawdown
)


# Download market data
def load_market_data():

    print("\nDownloading Market Data...")

    df = yf.download(
        MARKET_SYMBOL,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=True,
        progress=False
    )

    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    return df


# Feature engineering
def prepare_dataset(df):

    print("Creating Features...")

    df = create_features(df)

    df = add_rsi(df)

    df = add_macd(df)

    df = add_bollinger_bands(df)

    return df


# Prepare ML dataset
def prepare_ml_data(df):

    features = [
        "Returns",
        "MA_10",
        "MA_20",
        "Volatility",
        "Momentum",
        "Lag_1",
        "Lag_2",
        "RSI",
        "MACD",
        "MACD_Signal"
    ]

    X = df[features].values

    y = df["Target"].values

    return X, y


def main():

    print("="*60)
    print("QUANTEDGE")
    print("="*60)

    df = load_market_data()

    print(f"Rows : {len(df)}")

    df = prepare_dataset(df)

    regime = run_regime_detection(df)

    df = regime["data"]

    print("\nTraining Models...")

    X, y = prepare_ml_data(df)

    model_results = run_models(
        X,
        y
    )

    predictions = model_results["predictions"]

    rf = predictions["rf"]

    ann = predictions["ann"]

    lstm = predictions["lstm"]

    transformer = predictions["transformer"]

    min_len = min(
        len(rf),
        len(ann),
        len(lstm),
        len(transformer)
    )

    rf = rf[:min_len]
    ann = ann[:min_len]
    lstm = lstm[:min_len]
    transformer = transformer[:min_len]

    df = df.iloc[-min_len:].copy()

    print("Generating Signals...")

    signal_results = run_signal_generator(
        rf,
        ann,
        lstm,
        transformer,
        df["Close"].values
    )

    df = run_strategy(
        df,
        rf,
        ann,
        lstm,
        transformer
    )

    print("Optimizing Portfolio...")

    returns = df["Strategy_Return"].dropna()

    mean_returns = returns.to_frame().mean()

    covariance = returns.to_frame().cov()

    portfolio = optimize_portfolio(
        mean_returns,
        covariance
    )

    print("Running Backtest...")

    results = run_backtest(df)

    metrics = results["metrics"]

    statistics = results["statistics"]

    print_metrics(metrics)

    print("\nTrade Statistics")

    print("-"*40)

    for k, v in statistics.items():
        print(f"{k:<20}: {v}")

    print("\nPortfolio Allocation")

    print("-"*40)

    print("Maximum Sharpe Weights")

    print(portfolio["max_sharpe"])

    print("\nMinimum Variance Weights")

    print(portfolio["min_variance"])

    print("\nGenerating Charts...")

    plot_equity_curve(results["data"])

    plot_drawdown(results["data"])

    print("\nFinal Portfolio Value")

    print(
        f"₹ {results['data']['Cumulative_Strategy'].iloc[-1] * INITIAL_CAPITAL:,.2f}"
    )

    print("\nCompleted Successfully")


if __name__ == "__main__":
    main()
