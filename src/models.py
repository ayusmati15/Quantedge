from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score


def train_models(df):

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

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        shuffle=False
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    # Logistic Regression

    model_lr = LogisticRegression(
        max_iter=1000
    )

    model_lr.fit(
        X_train_scaled,
        y_train
    )

    y_pred_lr = model_lr.predict(
        X_test_scaled
    )

    acc_lr = accuracy_score(
        y_test,
        y_pred_lr
    )

    # Decision Tree

    model_dt = DecisionTreeClassifier(
        max_depth=5
    )

    model_dt.fit(
        X_train,
        y_train
    )

    y_pred_dt = model_dt.predict(
        X_test
    )

    acc_dt = accuracy_score(
        y_test,
        y_pred_dt
    )

    # Random Forest

    model_rf = RandomForestClassifier(
        n_estimators=100
    )

    model_rf.fit(
        X_train,
        y_train
    )

    y_pred_rf = model_rf.predict(
        X_test
    )

    acc_rf = accuracy_score(
        y_test,
        y_pred_rf
    )

    return {
        "Logistic Regression": acc_lr,
        "Decision Tree": acc_dt,
        "Random Forest": acc_rf
    }