import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from config import *

# Prepare dataset
def prepare_data(df):

    features=[
        "MA_10",
        "MA_20",
        "Volatility",
        "Momentum",
        "Lag_1",
        "Lag_2",
        "RSI",
        "MACD",
        "MACD_Signal"
    ]

    X=df[features]

    y=df["Target"]

    return X,y


# Train/Test Split
def split_dataset(X,y):

    return train_test_split(
        X,
        y,
        test_size=1-TRAIN_RATIO,
        shuffle=False
    )


# Scaling
def scale_data(X_train,X_test):

    scaler=StandardScaler()

    X_train=scaler.fit_transform(X_train)

    X_test=scaler.transform(X_test)

    return X_train,X_test,scaler


# Logistic Regression
def train_logistic(X_train,y_train):

    model=LogisticRegression(
        max_iter=1000,
        random_state=SEED
    )

    model.fit(X_train,y_train)

    return model


# Decision Tree
def train_decision_tree(X_train,y_train):

    model=DecisionTreeClassifier(
        max_depth=6,
        random_state=SEED
    )

    model.fit(X_train,y_train)

    return model


# Random Forest
def train_random_forest(X_train,y_train):

    model=RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        max_depth=RF_MAX_DEPTH,
        min_samples_split=RF_MIN_SAMPLES_SPLIT,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        random_state=SEED
    )

    model.fit(X_train,y_train)

    return model


# Prediction
def predict(model,X):

    return model.predict(X)


# Probability
def predict_probability(model,X):

    return model.predict_proba(X)[:,1]


# Save
def save_model(model,name):

    joblib.dump(
        model,
        f"{MODEL_DIRECTORY}/{name}"
    )


# Load
def load_model(name):

    return joblib.load(
        f"{MODEL_DIRECTORY}/{name}"
    )


# Evaluation
def evaluate_model(model,X_test,y_test):

    pred=model.predict(X_test)

    acc=accuracy_score(y_test,pred)

    precision=precision_score(
        y_test,
        pred,
        zero_division=0
    )

    recall=recall_score(
        y_test,
        pred,
        zero_division=0
    )

    f1=f1_score(
        y_test,
        pred,
        zero_division=0
    )

    cm=confusion_matrix(
        y_test,
        pred
    )

    return{
        "Accuracy":acc,
        "Precision":precision,
        "Recall":recall,
        "F1 Score":f1,
        "Confusion Matrix":cm
    }


# Compare models
def compare_models(models,X_test,y_test):

    results={}

    for name,model in models.items():

        results[name]=evaluate_model(
            model,
            X_test,
            y_test
        )

    return results


# Feature Importance
def feature_importance(model,feature_names):

    if not hasattr(model,"feature_importances_"):
        return None

    importance=pd.DataFrame({

        "Feature":feature_names,

        "Importance":model.feature_importances_

    })

    importance.sort_values(
        "Importance",
        ascending=False,
        inplace=True
    )

    return importance


# Train all classical ML models
def train_all_models(df):

    X,y=prepare_data(df)

    X_train,X_test,y_train,y_test=split_dataset(X,y)

    X_train_scaled,X_test_scaled,scaler=scale_data(
        X_train,
        X_test
    )

    lr=train_logistic(
        X_train_scaled,
        y_train
    )

    dt=train_decision_tree(
        X_train,
        y_train
    )

    rf=train_random_forest(
        X_train,
        y_train
    )

    models={

        "Logistic Regression":lr,

        "Decision Tree":dt,

        "Random Forest":rf

    }

    return{
        "models":models,
        "scaler":scaler,
        "X_train":X_train,
        "X_test":X_test,
        "X_train_scaled":X_train_scaled,
        "X_test_scaled":X_test_scaled,
        "y_train":y_train,
        "y_test":y_test
    }


# Predict all models
def predict_all_models(results):

    models=results["models"]

    predictions={}

    predictions["Logistic Regression"]=predict(
        models["Logistic Regression"],
        results["X_test_scaled"]
    )

    predictions["Decision Tree"]=predict(
        models["Decision Tree"],
        results["X_test"]
    )

    predictions["Random Forest"]=predict(
        models["Random Forest"],
        results["X_test"]
    )

    probabilities={}

    probabilities["Logistic Regression"]=predict_probability(
        models["Logistic Regression"],
        results["X_test_scaled"]
    )

    probabilities["Decision Tree"]=predict_probability(
        models["Decision Tree"],
        results["X_test"]
    )

    probabilities["Random Forest"]=predict_probability(
        models["Random Forest"],
        results["X_test"]
    )

    return predictions,probabilities


# Print results
def print_results(results):

    print("\nModel Performance")

    print("-"*50)

    for model,metrics in results.items():

        print(f"\n{model}")

        print(f"Accuracy : {metrics['Accuracy']:.4f}")

        print(f"Precision: {metrics['Precision']:.4f}")

        print(f"Recall   : {metrics['Recall']:.4f}")

        print(f"F1 Score : {metrics['F1 Score']:.4f}")
