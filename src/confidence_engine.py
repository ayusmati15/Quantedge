import numpy as np
import pandas as pd
from config import *

def confidence(prob):
    return abs(prob - 0.5) * 2

def confidence_label(score):
    if score >= 0.9:
        return "Very High"
    if score >= 0.75:
        return "High"
    if score >= 0.6:
        return "Medium"
    return "Low"

def filter_signals(probs):
    signals = []

    for p in probs:
        c = confidence(p)

        if c < CONFIDENCE_THRESHOLD:
            signals.append("HOLD")

        elif p >= BUY_THRESHOLD:
            signals.append("BUY")

        elif p <= SELL_THRESHOLD:
            signals.append("SELL")

        else:
            signals.append("HOLD")

    return signals

def confidence_dataframe(probabilities):
    df = pd.DataFrame()

    df["Probability"] = probabilities
    df["Confidence"] = [confidence(i) for i in probabilities]
    df["Label"] = [confidence_label(i) for i in df["Confidence"]]
    df["Signal"] = filter_signals(probabilities)

    return df

def summary(df):

    print("\nConfidence Summary")

    print(df["Label"].value_counts())

    print("\nSignal Summary")

    print(df["Signal"].value_counts())

def average_confidence(df):
    return df["Confidence"].mean()

def high_confidence_only(df):
    return df[df["Confidence"] >= CONFIDENCE_THRESHOLD]

def confidence_statistics(df):

    return {
        "Average": df["Confidence"].mean(),
        "Maximum": df["Confidence"].max(),
        "Minimum": df["Confidence"].min(),
        "Std": df["Confidence"].std()
    }

def export_confidence(df):

    path = REPORT_DIRECTORY + "/confidence_report.csv"

    df.to_csv(path,index=False)

    print("Saved:",path)

def run_confidence_engine(probabilities):

    df = confidence_dataframe(probabilities)

    summary(df)

    export_confidence(df)

    return df
