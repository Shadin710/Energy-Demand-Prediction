from src.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features,
    remove_null_rows
)


def run_feature_pipeline(df):

    df = create_time_features(df)

    df = create_lag_features(df)

    df = create_rolling_features(df)

    df = remove_null_rows(df)

    return df