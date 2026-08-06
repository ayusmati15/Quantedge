import matplotlib.pyplot as plt
import numpy as np

# Equity Curve
def plot_equity_curve(df):

    plt.figure(figsize=(13,6))

    plt.plot(
        df.index,
        df["Cumulative_Market"],
        label="Market",
        linewidth=2
    )

    plt.plot(
        df.index,
        df["Cumulative_Strategy"],
        label="Strategy",
        linewidth=2
    )

    plt.fill_between(
        df.index,
        df["Cumulative_Market"],
        df["Cumulative_Strategy"],
        alpha=0.15
    )

    plt.title("Portfolio vs Market")
    plt.xlabel("Date")
    plt.ylabel("Growth")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Drawdown
def plot_drawdown(df):

    drawdown=(
        df["Cumulative_Strategy"]/
        df["Cumulative_Strategy"].cummax()-1
    )

    plt.figure(figsize=(13,4))

    plt.plot(
        df.index,
        drawdown,
        linewidth=2
    )

    plt.fill_between(
        df.index,
        drawdown,
        0,
        alpha=0.3
    )

    plt.title("Strategy Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Buy/Sell Signals
def plot_signals(df):

    plt.figure(figsize=(14,6))

    plt.plot(
        df.index,
        df["Close"],
        label="Close",
        linewidth=2
    )

    buy=df[df["Signal"]=="BUY"]

    sell=df[df["Signal"]=="SELL"]

    plt.scatter(
        buy.index,
        buy["Close"],
        marker="^",
        s=90,
        label="BUY"
    )

    plt.scatter(
        sell.index,
        sell["Close"],
        marker="v",
        s=90,
        label="SELL"
    )

    plt.title("Trading Signals")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Confidence
def plot_confidence(df):

    plt.figure(figsize=(13,4))

    plt.plot(
        df.index,
        df["Confidence"],
        linewidth=2
    )

    plt.axhline(
        0.7,
        linestyle="--"
    )

    plt.title("Model Confidence")
    plt.xlabel("Date")
    plt.ylabel("Confidence")
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Portfolio Allocation
def plot_allocation(weights):

    plt.figure(figsize=(7,7))

    plt.pie(
        weights,
        autopct="%1.1f%%",
        startangle=90
    )

    plt.title("Portfolio Allocation")
    plt.tight_layout()


# Return Distribution
def plot_returns(df):

    plt.figure(figsize=(10,5))

    plt.hist(
        df["Strategy_Return"],
        bins=40
    )

    plt.title("Strategy Return Distribution")
    plt.xlabel("Daily Return")
    plt.ylabel("Frequency")
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Rolling Sharpe
def plot_rolling_sharpe(df,window=60):

    rolling=(
        df["Strategy_Return"]
        .rolling(window)
        .mean()/
        (
            df["Strategy_Return"]
            .rolling(window)
            .std()+1e-9
        )
    )*np.sqrt(252)

    plt.figure(figsize=(13,5))

    plt.plot(
        df.index,
        rolling,
        linewidth=2
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.title("Rolling Sharpe Ratio")
    plt.xlabel("Date")
    plt.ylabel("Sharpe")
    plt.grid(alpha=0.3)
    plt.tight_layout()


# Show everything
def show_dashboard(df,weights):

    plot_equity_curve(df)

    plot_drawdown(df)

    plot_signals(df)

    plot_confidence(df)

    plot_returns(df)

    plot_rolling_sharpe(df)

    plot_allocation(weights)

    plt.show()
