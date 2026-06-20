import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:

    return pd.read_csv(file_path)


def convert_datetime(df: pd.DataFrame) -> pd.DataFrame:

    df["Settlement Date"] = pd.to_datetime(
        df["Settlement Date"],
        dayfirst=True
    )

    return df


def sort_by_date(df: pd.DataFrame) -> pd.DataFrame:

    return df.sort_values(
        by="Settlement Date"
    ).reset_index(drop=True)


def check_missing_values(
        df: pd.DataFrame
) -> pd.Series:

    return df.isnull().sum()