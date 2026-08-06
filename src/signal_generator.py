import numpy as np
from config import *

MODEL_WEIGHTS = {
    "rf": 0.25,
    "ann": 0.20,
    "lstm": 0.25,
    "transformer": 0.30
}

def normalize_probability(p):
    return np.clip(p, 0, 1)

def ensemble_probability(rf, ann, lstm, transformer):

    probs = {
        "rf": normalize_probability(rf),
        "ann": normalize_probability(ann),
        "lstm": normalize_probability(lstm),
        "transformer": normalize_probability(transformer)
    }

    return sum(
        probs[m] * MODEL_WEIGHTS[m]
        for m in MODEL_WEIGHTS
    )

def probability_to_signal(prob):

    if prob >= BUY_THRESHOLD:
        return "BUY"

    if prob <= SELL_THRESHOLD:
        return "SELL"

    return "HOLD"

def confidence_score(prob):
    return abs(prob - 0.5) * 2

def confidence_filter(signal, confidence):

    if confidence < CONFIDENCE_THRESHOLD:
        return "HOLD"

    return signal

def majority_vote(*signals):

    buy = signals.count("BUY")
    sell = signals.count("SELL")
    hold = signals.count("HOLD")

    if buy > max(sell, hold):
        return "BUY"

    if sell > max(buy, hold):
        return "SELL"

    return "HOLD"

def technical_confirmation(rsi, macd, signal):

    if rsi < 30 and macd > signal:
        return "BUY"

    if rsi > 70 and macd < signal:
        return "SELL"

    return "HOLD"

def final_signal(
    rf_prob,
    ann_prob,
    lstm_prob,
    transformer_prob,
    rsi,
    macd,
    macd_signal
):

    prob = ensemble_probability(
        rf_prob,
        ann_prob,
        lstm_prob,
        transformer_prob
    )

    ai = probability_to_signal(prob)

    tech = technical_confirmation(
        rsi,
        macd,
        macd_signal
    )

    conf = confidence_score(prob)

    ai = confidence_filter(ai, conf)

    if ai == tech:
        return ai, prob, conf

    return "HOLD", prob, conf

import pandas as pd

def position_size(confidence, capital=INITIAL_CAPITAL):
    risk = 0.02
    return capital * risk * confidence

def stop_loss(price, pct=0.02):
    return price * (1 - pct)

def take_profit(price, pct=0.04):
    return price * (1 + pct)

def create_trade(signal, price, confidence):

    if signal == "HOLD":
        return None

    return {
        "Signal": signal,
        "Entry": price,
        "StopLoss": stop_loss(price),
        "TakeProfit": take_profit(price),
        "PositionSize": position_size(confidence)
    }

def portfolio_signals(probabilities, prices):

    trades = []

    for p, price in zip(probabilities, prices):

        signal = probability_to_signal(p)

        conf = confidence_score(p)

        signal = confidence_filter(signal, conf)

        trade = create_trade(signal, price, conf)

        trades.append(trade)

    return trades

def signal_statistics(signals):

    s = pd.Series(signals)

    return {
        "BUY": int((s == "BUY").sum()),
        "SELL": int((s == "SELL").sum()),
        "HOLD": int((s == "HOLD").sum()),
        "Total": len(s)
    }

def export_signals(signals, filename="signals.csv"):

    pd.DataFrame(signals).to_csv(
        REPORT_DIRECTORY + "/" + filename,
        index=False
    )

def run_signal_generator(
    rf_probs,
    ann_probs,
    lstm_probs,
    transformer_probs,
    prices
):

    probabilities = []

    signals = []

    confidences = []

    for rf, ann, lstm, tr in zip(
        rf_probs,
        ann_probs,
        lstm_probs,
        transformer_probs
    ):

        p = ensemble_probability(
            rf,
            ann,
            lstm,
            tr
        )

        probabilities.append(p)

        signal = probability_to_signal(p)

        conf = confidence_score(p)

        signal = confidence_filter(signal, conf)

        signals.append(signal)

        confidences.append(conf)

    trades = portfolio_signals(
        probabilities,
        prices
    )

    export_signals(trades)

    stats = signal_statistics(signals)

    return {
        "probabilities": probabilities,
        "signals": signals,
        "confidence": confidences,
        "trades": trades,
        "statistics": stats
    }


if __name__ == "__main__":

    print("Signal Generator Loaded")
