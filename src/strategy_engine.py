import pandas as pd

from regime_detection import detect_regime
from signal_generator import (
    ensemble_probability,
    probability_to_signal,
    confidence_score,
    confidence_filter
)


# Generate trading signals
def generate_strategy(
    df,
    rf_probs,
    ann_probs,
    lstm_probs,
    transformer_probs
):

    df = detect_regime(df)

    probabilities = []
    signals = []
    confidence = []

    for i in range(len(df)):

        p = ensemble_probability(
            rf_probs[i],
            ann_probs[i],
            lstm_probs[i],
            transformer_probs[i]
        )

        s = probability_to_signal(p)

        c = confidence_score(p)

        regime = df["Regime"].iloc[i]

        if regime == "bearish" and s == "BUY":
            s = "HOLD"

        elif regime == "bullish" and s == "SELL":
            s = "HOLD"

        s = confidence_filter(s, c)

        probabilities.append(p)
        confidence.append(c)
        signals.append(s)

    df["Probability"] = probabilities
    df["Confidence"] = confidence
    df["Signal"] = signals

    return df


# Position values
def create_positions(df):

    position = []

    current = 0

    for signal in df["Signal"]:

        if signal == "BUY":
            current = 1

        elif signal == "SELL":
            current = -1

        position.append(current)

    df["Position"] = position

    return df


# Strategy returns
def strategy_returns(df):

    df = df.copy()

    df["Market_Return"] = df["Close"].pct_change().fillna(0)

    df["Strategy_Return"] = (
        df["Position"].shift(1).fillna(0)
        * df["Market_Return"]
    )

    df["Cumulative_Return"] = (
        1 + df["Strategy_Return"]
    ).cumprod()

    return df


# Complete pipeline
def run_strategy(
    df,
    rf_probs,
    ann_probs,
    lstm_probs,
    transformer_probs
):

    df = generate_strategy(
        df,
        rf_probs,
        ann_probs,
        lstm_probs,
        transformer_probs
    )

    df = create_positions(df)

    df = strategy_returns(df)

    return df


if __name__ == "__main__":

    print("Strategy Engine Loaded")
