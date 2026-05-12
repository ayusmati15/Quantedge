import matplotlib.pyplot as plt
def plot_equity_curve(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Cumulative_Market"],
        label="Market"
    )

    plt.plot(
        df["Cumulative_Strategy"],
        label="Strategy"
    )

    plt.title("Equity Curve Comparison")
    plt.xlabel("Time")
    plt.ylabel("Growth of ₹1")
    plt.legend()
    plt.grid()
    plt.show()


def plot_drawdown(df):

    drawdown = (
        df["Cumulative_Strategy"]
        / df["Cumulative_Strategy"].cummax()
        - 1
    )

    plt.figure(figsize=(12, 4))

    plt.plot(drawdown)
    plt.title("Strategy Drawdown")
    plt.xlabel("Time")
    plt.ylabel("Drawdown")
    plt.grid()
    plt.show()