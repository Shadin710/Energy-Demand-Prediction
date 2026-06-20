import pandas as pd

from src.feature_engineering import (
    create_time_features,
    create_lag_features,
    create_rolling_features
)

def test_time_features():

    df = pd.DataFrame({
        "Settlement Date":
        pd.to_datetime(
            ["2026-06-17 12:30"]
        )
    })

    result = create_time_features(df)

    assert "hour" in result.columns
    assert "weekday" in result.columns

def test_lag_features():

    df = pd.DataFrame({
        "Scheduled Demand (MW)":
        range(500)
    })

    result = create_lag_features(df)

    assert "lag_1" in result.columns
    assert "lag_12" in result.columns
    assert "lag_288" in result.columns
def test_rolling_features():

    df = pd.DataFrame({
        "Scheduled Demand (MW)":
        range(500)
    })

    result = create_rolling_features(df)

    assert "rolling_mean_12" in result.columns