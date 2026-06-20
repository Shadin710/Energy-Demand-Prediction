from src.pipeline import (
    run_pipeline
)


def main():

    df = run_pipeline(
        "data/electricity_demand.csv"
    )

    print(df.head())

    print(df.shape)


if __name__ == "__main__":
    main()