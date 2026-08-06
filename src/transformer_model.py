
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.layers import (
    Input,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention,
    GlobalAveragePooling1D
)

from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler

from config import *
from utils import *

# POSITIONAL ENCODING

class PositionalEncoding(tf.keras.layers.Layer):

    def __init__(self, sequence_length, embed_dim):
        super().__init__()

        self.position_embedding = tf.keras.layers.Embedding(
            input_dim=sequence_length,
            output_dim=embed_dim
        )

    def call(self, inputs):

        positions = tf.range(
            start=0,
            limit=tf.shape(inputs)[1],
            delta=1
        )

        position_embeddings = self.position_embedding(positions)

        return inputs + position_embeddings

class TransformerEncoder(tf.keras.layers.Layer):

    def __init__(
        self,
        embed_dim,
        num_heads,
        feed_forward_dim,
        dropout=0.1
    ):

        super().__init__()

        self.attention = MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )

        self.feed_forward = tf.keras.Sequential(
            [
                Dense(feed_forward_dim, activation="relu"),
                Dense(embed_dim)
            ]
        )

        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training=False):

        attention_output = self.attention(
            inputs,
            inputs
        )

        attention_output = self.dropout1(
            attention_output,
            training=training
        )

        out1 = self.norm1(
            inputs + attention_output
        )

        feed_output = self.feed_forward(out1)

        feed_output = self.dropout2(
            feed_output,
            training=training
        )

        return self.norm2(
            out1 + feed_output
        )

def build_transformer(
    sequence_length=TRANSFORMER_SEQUENCE_LENGTH,
    num_features=1,
    embed_dim=TRANSFORMER_EMBED_DIM,
    num_heads=TRANSFORMER_HEADS,
    ff_dim=TRANSFORMER_FEED_FORWARD,
    num_layers=TRANSFORMER_LAYERS,
    dropout=TRANSFORMER_DROPOUT
):

    inputs = Input(
        shape=(sequence_length, num_features)
    )

    x = Dense(embed_dim)(inputs)

    x = PositionalEncoding(
        sequence_length,
        embed_dim
    )(x)

    for _ in range(num_layers):

        x = TransformerEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            feed_forward_dim=ff_dim,
            dropout=dropout
        )(x)

    x = GlobalAveragePooling1D()(x)

    x = Dropout(dropout)(x)

    x = Dense(
        128,
        activation="relu"
    )(x)

    x = Dropout(dropout)(x)

    outputs = Dense(
        1,
        activation="sigmoid"
    )(x)

    model = Model(
        inputs,
        outputs
    )

    optimizer = Adam(
        learning_rate=ANN_LEARNING_RATE
    )

    model.compile(

        optimizer=optimizer,

        loss="binary_crossentropy",

        metrics=[
            "accuracy"
        ]
    )

    return model


def print_transformer_summary(model):

    print_header("Transformer Architecture")

    model.summary()

def initialize_transformer(num_features):

    model = build_transformer(

        sequence_length=TRANSFORMER_SEQUENCE_LENGTH,

        num_features=num_features,

        embed_dim=TRANSFORMER_EMBED_DIM,

        num_heads=TRANSFORMER_HEADS,

        ff_dim=TRANSFORMER_FEED_FORWARD,

        num_layers=TRANSFORMER_LAYERS,

        dropout=TRANSFORMER_DROPOUT
    )

    return model

# SEQUENCE CREATION

def create_sequences(
    features,
    targets,
    sequence_length=TRANSFORMER_SEQUENCE_LENGTH
):
    """
    Convert tabular data into sequences for Transformer.
    """

    X = []
    y = []

    for i in range(len(features) - sequence_length):

        X.append(
            features[i:i + sequence_length]
        )

        y.append(
            targets[i + sequence_length]
        )

    return np.array(X), np.array(y)
# FEATURE SCALING


def scale_features(df, feature_columns):

    scaler = MinMaxScaler()

    scaled = scaler.fit_transform(
        df[feature_columns]
    )

    return scaled, scaler

# TARGET CREATION

def create_binary_target(df):

    target = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    return target.values
# DATA PREPARATION


def prepare_transformer_dataset(
    dataframe,
    feature_columns
):
    """
    Complete preprocessing pipeline.
    """

    df = dataframe.copy()

    df = df.dropna()

    scaled_features, scaler = scale_features(
        df,
        feature_columns
    )

    targets = create_binary_target(df)

    X, y = create_sequences(
        scaled_features,
        targets,
        TRANSFORMER_SEQUENCE_LENGTH
    )

    return X, y, scaler

# TRAIN TEST SPLIT

def split_transformer_data(X, y):

    split = int(
        len(X) * TRAIN_RATIO
    )

    X_train = X[:split]
    X_test = X[split:]

    y_train = y[:split]
    y_test = y[split:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

# TRAINING

@execution_time
def train_transformer(
    model,
    X_train,
    y_train
):

    early_stop = EarlyStopping(

        monitor="val_loss",

        patience=5,

        restore_best_weights=True
    )

    history = model.fit(

        X_train,

        y_train,

        validation_split=0.2,

        epochs=TRANSFORMER_EPOCHS,

        batch_size=TRANSFORMER_BATCH_SIZE,

        callbacks=[early_stop],

        verbose=1
    )

    return history


# ==========================================================
# COMPLETE TRAINING PIPELINE


def fit_transformer(
    dataframe,
    feature_columns
):
    """
    End-to-end training pipeline.
    """

    print_header(
        "Preparing Transformer Dataset"
    )

    X, y, scaler = prepare_transformer_dataset(

        dataframe,

        feature_columns
    )

    (
        X_train,
        X_test,
        y_train,
        y_test

    ) = split_transformer_data(
        X,
        y
    )

    model = initialize_transformer(
        num_features=len(feature_columns)
    )

    history = train_transformer(
        model,
        X_train,
        y_train
    )

    return {

        "model": model,

        "history": history,

        "scaler": scaler,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test
    }

# TRAINING HISTORY

def plot_training_history(history):

    plt.figure(figsize=(10,5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.title(
        "Transformer Training Loss"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Loss")

    plt.legend()

    if SAVE_PLOTS:

        save_plot(
            "transformer_loss.png"
        )

    plt.show()


def plot_accuracy(history):

    plt.figure(figsize=(10,5))

    plt.plot(
        history.history["accuracy"],
        label="Training Accuracy"
    )

    plt.plot(
        history.history["val_accuracy"],
        label="Validation Accuracy"
    )

    plt.title(
        "Transformer Accuracy"
    )

    plt.xlabel("Epoch")

    plt.ylabel("Accuracy")

    plt.legend()

    if SAVE_PLOTS:

        save_plot(
            "transformer_accuracy.png"
        )

    plt.show()

# PREDICTION


def predict_probabilities(
    model,
    X_test
):
    """
    Returns prediction probabilities.
    """

    probabilities = model.predict(
        X_test,
        verbose=0
    ).flatten()

    return probabilities


def predict_classes(
    model,
    X_test,
    threshold=0.5
):

    probabilities = predict_probabilities(
        model,
        X_test
    )

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return predictions
# SIGNAL GENERATION


def generate_signals(
    probabilities,
    buy_threshold=BUY_THRESHOLD,
    sell_threshold=SELL_THRESHOLD
):
    """
    BUY  -> Probability >= BUY_THRESHOLD
    SELL -> Probability <= SELL_THRESHOLD
    HOLD -> Otherwise
    """

    signals = []

    for probability in probabilities:

        if probability >= buy_threshold:

            signals.append("BUY")

        elif probability <= sell_threshold:

            signals.append("SELL")

        else:

            signals.append("HOLD")

    return np.array(signals)

# MODEL EVALUATION


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    classification_report
)


def evaluate_transformer(
    model,
    X_test,
    y_test
):

    predictions = predict_classes(
        model,
        X_test
    )

    probabilities = predict_probabilities(
        model,
        X_test
    )

    metrics = {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions
            ),

        "Recall":
            recall_score(
                y_test,
                predictions
            ),

        "F1 Score":
            f1_score(
                y_test,
                predictions
            ),

        "ROC AUC":
            roc_auc_score(
                y_test,
                probabilities
            )

    }

    print_header(
        "Transformer Evaluation"
    )

    for key, value in metrics.items():

        print(f"{key:15s}: {value:.4f}")

    print("\nClassification Report\n")

    print(

        classification_report(

            y_test,

            predictions

        )

    )

    return metrics


# ==========================================================
# CONFUSION MATRIX
# ==========================================================

def plot_confusion(
    model,
    X_test,
    y_test
):

    predictions = predict_classes(
        model,
        X_test
    )

    cm = confusion_matrix(
        y_test,
        predictions
    )

    plt.figure(figsize=(6,6))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.title(
        "Transformer Confusion Matrix"
    )

    plt.xlabel(
        "Predicted"
    )

    plt.ylabel(
        "Actual"
    )

    for i in range(cm.shape[0]):

        for j in range(cm.shape[1]):

            plt.text(
                j,
                i,
                str(cm[i,j]),
                ha="center",
                va="center"
            )

    plt.colorbar()

    if SAVE_PLOTS:

        save_plot(
            "transformer_confusion.png"
        )

    plt.show()

# SAVE / LOAD


def save_transformer(
    model
):

    save_keras_model(

        model,

        TRANSFORMER_MODEL_NAME

    )


def load_transformer():

    return load_keras_model(

        TRANSFORMER_MODEL_NAME

    )

# COMPLETE PIPELINE

def run_transformer_pipeline(
    dataframe,
    feature_columns
):

    print_header(
        "Running Transformer Pipeline"
    )

    results = fit_transformer(

        dataframe,

        feature_columns

    )

    model = results["model"]

    history = results["history"]

    X_test = results["X_test"]

    y_test = results["y_test"]

    plot_training_history(
        history
    )

    plot_accuracy(
        history
    )

    metrics = evaluate_transformer(

        model,

        X_test,

        y_test

    )

    probabilities = predict_probabilities(

        model,

        X_test

    )

    signals = generate_signals(

        probabilities

    )

    save_transformer(
        model
    )

    return {

        "model": model,

        "metrics": metrics,

        "probabilities": probabilities,

        "signals": signals,

        "history": history,

        "scaler": results["scaler"]

    }

# STANDALONE TEST
if __name__ == "__main__":

    print_header(
        "Transformer Module Loaded Successfully"
    )
