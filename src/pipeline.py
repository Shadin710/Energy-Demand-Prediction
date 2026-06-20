from src.data_loader import (
    load_data,
    convert_datetime,
    sort_by_date
)

from src.feature_pipeline import (
    run_feature_pipeline
)


def run_pipeline(file_path):

    df = load_data(file_path)

    df = convert_datetime(df)

    df = sort_by_date(df)

    df = run_feature_pipeline(df)

    return df