import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier


def train_ai_model(df):

    features = df[
        [
            "MA_10",
            "MA_20",
            "Volatility",
            "Momentum",
            "Lag_1",
            "Lag_2"
        ]
    ]

    target = df["Target"]

    split = int(len(df) * 0.8)

    X_train = features[:split]

    X_test = features[split:]

    y_train = target[:split]

    y_test = target[split:]

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = (predictions == y_test).mean()

    df_test = df.iloc[split:].copy()

    df_test["AI_Signal"] = predictions

    # SAFE NUMPY CONVERSION (avoids pandas bugs)
    returns = df_test["Close"].pct_change().fillna(0).values

    signals = df_test["AI_Signal"].shift(1).fillna(0).values

    df_test["AI_Return"] = signals * returns

    df_test["AI_Cumulative"] = (
        1 + df_test["AI_Return"]
    ).cumprod()

    return model, accuracy, df_test