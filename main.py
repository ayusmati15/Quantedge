import yfinance as yf

import matplotlib.pyplot as plt

from src.feature_engineering import create_features

from src.technical_indicators import (
    add_rsi,
    add_macd,
    add_bollinger_bands
)

from src.strategy_engine import (
    hybrid_strategy,
    calculate_strategy_returns
)

from src.ai_model import train_ai_model


def main():

    # DOWNLOAD MARKET DATA
    df = yf.download(
        "^NSEI",
        start="2020-01-01",
        auto_adjust=True
    )

    # FIX MULTIINDEX COLUMNS
    if hasattr(df.columns, "levels"):
        df.columns = df.columns.get_level_values(0)

    # DEBUG PRINT
    print("\nDATAFRAME COLUMNS:\n")
    print(df.columns)

    # FEATURE ENGINEERING
    df = create_features(df)

    # TECHNICAL INDICATORS
    df = add_rsi(df)

    df = add_macd(df)

    df = add_bollinger_bands(df)

    # PRINT INDICATORS
    print("\n=========== TECHNICAL INDICATORS ===========\n")

    print(
        df[
            [
                "RSI",
                "MACD",
                "MACD_Signal",
                "BB_Upper",
                "BB_Lower"
            ]
        ].tail()
    )

    print("\n============================================\n")

    # STRATEGY ENGINE
    df = hybrid_strategy(df)

    df = calculate_strategy_returns(df)

    # PRINT STRATEGY SIGNALS
    print("\n=========== STRATEGY SIGNALS ===========\n")

    print(
        df[
            [
                "RSI_Signal",
                "MACD_Signal_Strategy",
                "Hybrid_Signal"
            ]
        ].tail()
    )

    print("\n========================================\n")

    # AI MODEL
    model, acc, df_test = train_ai_model(df)

    # PRINT AI RESULTS
    print("\n================ AI MODEL ================\n")

    print(f"Accuracy: {acc:.4f}")

    print("\n==========================================\n")

    # HYBRID STRATEGY GRAPH
    plt.figure(figsize=(12, 6))

    plt.plot(
        df["Cumulative_Strategy"],
        label="Hybrid Strategy"
    )

    plt.title("Hybrid Trading Strategy")

    plt.xlabel("Time")

    plt.ylabel("Cumulative Returns")

    plt.legend()

    plt.grid()

    # AI STRATEGY GRAPH
    plt.figure(figsize=(12, 6))

    plt.plot(
        df_test["AI_Cumulative"],
        label="AI Strategy"
    )

    plt.title("AI Trading Strategy Performance")

    plt.xlabel("Time")

    plt.ylabel("Cumulative Returns")

    plt.legend()

    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()