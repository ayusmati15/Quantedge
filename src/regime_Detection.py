import numpy as np
import pandas as pd
from config import *

# Detect market regime
def detect_regime(df):

    df = df.copy()

    df["Return"] = df["Close"].pct_change()

    df["MA20"] = df["Close"].rolling(REGIME_WINDOW).mean()

    df["Volatility"] = df["Return"].rolling(REGIME_WINDOW).std()

    regimes = []

    for _, row in df.iterrows():

        if pd.isna(row["MA20"]):
            regimes.append(UNKNOWN)

        elif row["Close"] > row["MA20"] and row["Volatility"] < 0.02:
            regimes.append(BULLISH)

        elif row["Close"] < row["MA20"] and row["Volatility"] < 0.02:
            regimes.append(BEARISH)

        else:
            regimes.append(SIDEWAYS)

    df["Regime"] = regimes

    return df


# Current market regime
def current_regime(df):

    return df["Regime"].iloc[-1]


# Regime statistics
def regime_statistics(df):

    counts = df["Regime"].value_counts()

    return {
        "Bullish": int(counts.get(BULLISH, 0)),
        "Bearish": int(counts.get(BEARISH, 0)),
        "Sideways": int(counts.get(SIDEWAYS, 0))
    }


# Trading weight based on regime
def regime_weight(regime):

    if regime == BULLISH:
        return 1.0

    if regime == BEARISH:
        return 0.5

    return 0.75


# Modify confidence using regime
def adjust_confidence(confidence, regime):

    return confidence * regime_weight(regime)


# Regime recommendation
def regime_signal(regime):

    if regime == BULLISH:
        return "LONG"

    if regime == BEARISH:
        return "SHORT"

    return "WAIT"


# Complete pipeline
def run_regime_detection(df):

    df = detect_regime(df)

    regime = current_regime(df)

    stats = regime_statistics(df)

    print("\nCurrent Regime :", regime)

    print(stats)

    return {
        "data": df,
        "current": regime,
        "statistics": stats
    }


if __name__ == "__main__":

    print("Regime Detection Loaded")
