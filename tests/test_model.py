import pandas as pd

from src.model import (
    prepare_features_and_target
)


def test_feature_target_split():

    df = pd.DataFrame({
        "Scheduled Demand (MW)":
        [1, 2, 3],

        "Feature":
        [4, 5, 6]
    })

    X, y = prepare_features_and_target(
        df
    )

    assert len(X.columns) == 1

    assert len(y) == 3