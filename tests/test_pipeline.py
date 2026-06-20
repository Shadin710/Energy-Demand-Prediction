from src.pipeline import (
    run_pipeline
)


def test_pipeline_runs():

    results = run_pipeline(
        "data/electricity_demand.csv"
    )

    assert "metrics" in results

    assert "model" in results