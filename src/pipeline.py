from src.data_loader import (
    load_data,
    convert_datetime,
    sort_by_date
)

from src.feature_pipeline import (
    run_feature_pipeline
)

from src.model_pipeline import (
    run_model_pipeline
)
from src.report_pipeline import (
    run_reporting_pipeline
)


def run_pipeline(file_path):

    df = load_data(file_path)

    df = convert_datetime(df)

    df = sort_by_date(df)

    df = run_feature_pipeline(df)

    results = run_model_pipeline(df)
    
    run_reporting_pipeline(
    df=df,
    model=results["model"],
    y_test=results["y_test"],
    predictions=results["predictions"],
    feature_names=results[
        "feature_names"
    ])

    return results