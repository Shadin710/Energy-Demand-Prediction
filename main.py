from src.pipeline import (
    run_pipeline
)


def main():

    results = run_pipeline(
        "data/electricity_demand.csv"
    )

    print("\nModel Metrics")

    for metric, value in results[
        "metrics"
    ].items():

        print(
            f"{metric}: {value:.4f}"
        )


if __name__ == "__main__":
    main()