import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    LSTM,
    Dropout
)
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(
            data[i - seq_length:i]
        )
        y.append(data[i])
    return np.array(X), np.array(y)


def train_ann_lstm(df):
    prices = df["Close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(
        prices
    )
    seq_length = 20
    X, y = create_sequences(
        scaled_data,
        seq_length
    )

    split = int(0.8 * len(X))
    X_train = X[:split]
    X_test = X[split:]
    y_train = y[:split]
    y_test = y[split:]
    # ANN MODEL

    X_train_ann = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1]
    )

    X_test_ann = X_test.reshape(
        X_test.shape[0],
        X_test.shape[1]
    )

    ann_model = Sequential()

    ann_model.add(
        Dense(
            64,
            activation="relu",
            input_shape=(seq_length,)
        )
    )

    ann_model.add(
        Dense(
            32,
            activation="relu"
        )
    )

    ann_model.add(
        Dense(1)
    )

    ann_model.compile(
        optimizer="adam",
        loss="mse"
    )

    ann_model.fit(
        X_train_ann,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    ann_loss = ann_model.evaluate(
        X_test_ann,
        y_test,
        verbose=0
    )

    # LSTM MODEL

    lstm_model = Sequential()

    lstm_model.add(
        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                seq_length,
                1
            )
        )
    )

    lstm_model.add(
        Dropout(0.2)
    )

    lstm_model.add(
        LSTM(32)
    )

    lstm_model.add(
        Dropout(0.2)
    )

    lstm_model.add(
        Dense(1)
    )

    lstm_model.compile(
        optimizer="adam",
        loss="mse"
    )

    lstm_model.fit(
        X_train,
        y_train,
        epochs=10,
        batch_size=32,
        verbose=0
    )

    lstm_loss = lstm_model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    return {
        "ANN Loss": ann_loss,
        "LSTM Loss": lstm_loss
    }