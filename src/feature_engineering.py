import pandas as pd


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["hour"] = df["Settlement Date"].dt.hour

    df["minute"] = df["Settlement Date"].dt.minute

    df["day"] = df["Settlement Date"].dt.day

    df["month"] = df["Settlement Date"].dt.month

    df["weekday"] = df["Settlement Date"].dt.weekday

    df["weekend"] = (
        df["weekday"] >= 5
    ).astype(int)

    return df
def create_lag_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    target = "Scheduled Demand (MW)"

    df["lag_1"] = df[target].shift(1)

    df["lag_12"] = df[target].shift(12)

    df["lag_288"] = df[target].shift(288)

    return df
def create_rolling_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    target = "Scheduled Demand (MW)"

    df["rolling_mean_12"] = (
        df[target]
        .rolling(window=12)
        .mean()
    )

    df["rolling_std_12"] = (
        df[target]
        .rolling(window=12)
        .std()
    )

    df["rolling_mean_288"] = (
        df[target]
        .rolling(window=288)
        .mean()
    )

    return df
def remove_null_rows(df: pd.DataFrame) -> pd.DataFrame:

    return df.dropna()