from src.data_loader import (
    load_data,
    convert_datetime,
    sort_by_date,
    check_missing_values
)


def run_data_pipeline(
        file_path: str
):

    df = load_data(file_path)

    df = convert_datetime(df)

    df = sort_by_date(df)

    missing_values = check_missing_values(df)

    return df, missing_values