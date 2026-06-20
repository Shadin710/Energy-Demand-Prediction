import pandas as pd
from xgboost import XGBRegressor
import joblib

TARGET = "Scheduled Demand (MW)"


def prepare_features_and_target(df: pd.DataFrame):

    X = df.drop(
        columns=[
            TARGET,
            "Settlement Date",
            "Type"
        ],
        errors="ignore"
    )

    y = df[TARGET]

    return X, y
def train_test_split_time_series(X,y,train_size=0.8):

    split_idx = int(len(X) * train_size)

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )

def train_model(X_train,y_train):

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train,y_train)

    return model
def make_predictions(model,X_test):

    predictions = model.predict(X_test)

    return predictions
def save_model(model,path):
    joblib.dump(model,path)