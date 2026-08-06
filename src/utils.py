"""
===========================================================
Quantedge Utility Functions
===========================================================

Reusable helper functions used throughout the project.

Author : Ayusmati Panda
Project : Quantedge
"""

import os
import random
import time
import pickle
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from config import *


# ==========================================================
# DIRECTORY MANAGEMENT
# ==========================================================

def create_project_directories():
    """
    Creates required project directories if they don't exist.
    """

    directories = [
        MODEL_DIRECTORY,
        PLOT_DIRECTORY,
        REPORT_DIRECTORY,
        DATA_DIRECTORY
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    if VERBOSE:
        print("✓ Project directories ready.")


# ==========================================================
# RANDOM SEED
# ==========================================================

def set_random_seed(seed=SEED):
    """
    Makes experiments reproducible.
    """

    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    if VERBOSE:
        print(f"✓ Random seed set to {seed}")


# ==========================================================
# TIMER
# ==========================================================

class Timer:

    def __init__(self):
        self.start_time = None

    def start(self):

        self.start_time = time.time()

    def stop(self):

        if self.start_time is None:
            return 0

        elapsed = time.time() - self.start_time

        print(f"Completed in {elapsed:.2f} seconds")

        return elapsed


# ==========================================================
# EXECUTION DECORATOR
# ==========================================================

def execution_time(func):

    def wrapper(*args, **kwargs):

        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()

        print(f"{func.__name__} : {end-start:.2f} sec")

        return result

    return wrapper


# ==========================================================
# LOGGING
# ==========================================================

def initialize_logger():

    logging.basicConfig(

        filename="quantedge.log",

        level=logging.INFO,

        format="%(asctime)s %(levelname)s %(message)s"
    )

    logging.info("========== NEW SESSION ==========")


def log(message):

    logging.info(message)

    if VERBOSE:
        print(message)


# ==========================================================
# MODEL SAVE / LOAD
# ==========================================================

def save_sklearn_model(model, filename):

    path = os.path.join(MODEL_DIRECTORY, filename)

    joblib.dump(model, path)

    log(f"Saved model -> {path}")


def load_sklearn_model(filename):

    path = os.path.join(MODEL_DIRECTORY, filename)

    return joblib.load(path)


def save_pickle(obj, filename):

    path = os.path.join(MODEL_DIRECTORY, filename)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    log(f"Saved pickle -> {path}")


def load_pickle(filename):

    path = os.path.join(MODEL_DIRECTORY, filename)

    with open(path, "rb") as f:

        return pickle.load(f)


# ==========================================================
# KERAS MODELS
# ==========================================================

def save_keras_model(model, filename):

    path = os.path.join(MODEL_DIRECTORY, filename)

    model.save(path)

    log(f"Saved keras model -> {path}")


def load_keras_model(filename):

    path = os.path.join(MODEL_DIRECTORY, filename)

    return tf.keras.models.load_model(path)


# ==========================================================
# DATAFRAME UTILITIES
# ==========================================================

def print_dataframe_info(df):

    print("\nShape :", df.shape)

    print("\nColumns")

    print(df.columns.tolist())

    print("\nMissing Values")

    print(df.isnull().sum())

    print("\nData Types")

    print(df.dtypes)


def missing_value_report(df):

    report = pd.DataFrame({

        "Missing": df.isnull().sum(),

        "Percent": 100 * df.isnull().mean()

    })

    return report


# ==========================================================
# PLOT SAVE
# ==========================================================

def save_plot(name):

    filename = os.path.join(PLOT_DIRECTORY, name)

    plt.tight_layout()

    plt.savefig(filename, dpi=DPI)

    log(f"Plot saved -> {filename}")


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_series(series):

    return (series-series.mean())/(series.std()+1e-8)


def min_max_scale(series):

    return (series-series.min())/(series.max()-series.min()+1e-8)


# ==========================================================
# REPORT
# ==========================================================

def save_metrics(metrics, filename="metrics.csv"):

    path = os.path.join(REPORT_DIRECTORY, filename)

    df = pd.DataFrame(metrics, index=[0])

    df.to_csv(path, index=False)

    log(f"Metrics saved -> {path}")


# ==========================================================
# GPU
# ==========================================================

def gpu_information():

    devices = tf.config.list_physical_devices()

    print("\nAvailable Devices")

    for device in devices:

        print(device)


# ==========================================================
# MEMORY
# ==========================================================

def memory_usage(df):

    memory = df.memory_usage(deep=True).sum()

    print(f"Memory Usage : {memory/1024**2:.2f} MB")


# ==========================================================
# SHUFFLE
# ==========================================================

def shuffle_dataframe(df):

    return df.sample(frac=1, random_state=SEED).reset_index(drop=True)


# ==========================================================
# TRAIN TEST SPLIT INDEX
# ==========================================================

def split_index(length, ratio=TRAIN_RATIO):

    split = int(length*ratio)

    return split


# ==========================================================
# CUMULATIVE RETURNS
# ==========================================================

def cumulative_returns(returns):

    return (1+returns).cumprod()


# ==========================================================
# DRAWDOWN
# ==========================================================

def calculate_drawdown(equity):

    running_max = np.maximum.accumulate(equity)

    drawdown = (equity-running_max)/running_max

    return drawdown


# ==========================================================
# PRINT SECTION
# ==========================================================

def print_header(title):

    print("\n")

    print("="*70)

    print(title)

    print("="*70)


# ==========================================================
# SYSTEM INFORMATION
# ==========================================================

def system_summary():

    print_header("QUANTEDGE")

    print("Models Directory :", MODEL_DIRECTORY)

    print("Plots Directory :", PLOT_DIRECTORY)

    print("Reports Directory :", REPORT_DIRECTORY)

    print("Initial Capital :", INITIAL_CAPITAL)

    print("Risk Free Rate :", RISK_FREE_RATE)

    print("Transaction Cost :", TRANSACTION_COST)

    print("Confidence Threshold :", CONFIDENCE_THRESHOLD)

    print("Transformer Epochs :", TRANSFORMER_EPOCHS)

    print("LSTM Epochs :", LSTM_EPOCHS)

    print("Random Forest Trees :", RF_N_ESTIMATORS)


# ==========================================================
# INITIALIZATION
# ==========================================================

def initialize_project():

    create_project_directories()

    set_random_seed()

    initialize_logger()

    system_summary()

    log("Project Initialized Successfully")
