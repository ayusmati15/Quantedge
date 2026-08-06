import numpy as np
import pandas as pd
from sklearn.base import clone
from config import *

# Create rolling windows
def create_windows(X, y):

    windows = []

    start = 0

    while start + WF_TRAIN_SIZE + WF_TEST_SIZE <= len(X):

        train_end = start + WF_TRAIN_SIZE

        test_end = train_end + WF_TEST_SIZE

        windows.append((
            X[start:train_end],
            y[start:train_end],
            X[train_end:test_end],
            y[train_end:test_end]
        ))

        start += WF_STEP_SIZE

    return windows


# Perform walk-forward validation
def walk_forward_validation(model, X, y):

    windows = create_windows(X, y)

    accuracies = []

    predictions = []

    actuals = []

    for X_train, y_train, X_test, y_test in windows:

        m = clone(model)

        m.fit(X_train, y_train)

        pred = m.predict(X_test)

        if pred.ndim > 1:
            pred = pred.ravel()

        pred = (pred >= 0.5).astype(int)

        accuracy = (pred == y_test).mean()

        accuracies.append(accuracy)

        predictions.extend(pred)

        actuals.extend(y_test)

    return {
        "accuracy": np.mean(accuracies),
        "window_accuracy": accuracies,
        "predictions": np.array(predictions),
        "actual": np.array(actuals)
    }


# Summary
def print_summary(results):

    print("\nWalk Forward Validation")

    print("-----------------------")

    print(f"Windows : {len(results['window_accuracy'])}")

    print(f"Average Accuracy : {results['accuracy']:.4f}")

    print(f"Best Accuracy : {max(results['window_accuracy']):.4f}")

    print(f"Worst Accuracy : {min(results['window_accuracy']):.4f}")


# Accuracy DataFrame
def accuracy_dataframe(results):

    return pd.DataFrame({
        "Window": range(1, len(results["window_accuracy"]) + 1),
        "Accuracy": results["window_accuracy"]
    })


# Export results
def export_results(results):

    df = accuracy_dataframe(results)

    path = REPORT_DIRECTORY + "/walk_forward_results.csv"

    df.to_csv(path, index=False)

    print("Saved:", path)


# Complete pipeline
def run_walk_forward(model, X, y):

    results = walk_forward_validation(model, X, y)

    print_summary(results)

    export_results(results)

    return results


if __name__ == "__main__":

    print("Walk Forward Module Loaded")
