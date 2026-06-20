import pandas as pd

from src.data_loader import (
    load_data,
    convert_datetime
)


def test_load_data():

    df = load_data(
        "data/electricity_demand.csv"
    )

    assert isinstance(
        df,
        pd.DataFrame
    )


def test_datetime_conversion():

    sample = pd.DataFrame(
        {
            "Settlement Date":
            ["17/06/2026 00:00"]
        }
    )

    result = convert_datetime(
        sample
    )

    assert pd.api.types.is_datetime64_any_dtype(
        result["Settlement Date"]
    )