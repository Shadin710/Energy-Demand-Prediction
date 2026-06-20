from src.pipeline import (
    run_data_pipeline
)


def main():

    df, missing_values = run_data_pipeline(
        "data/electricity_demand.csv"
    )

    print(df.head())

    print("\nMissing Values")

    print(missing_values)


if __name__ == "__main__":
    main()