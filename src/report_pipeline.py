from src.visualization import (
    plot_actual_vs_forecast,
    plot_feature_importance,
    plot_residuals,
    plot_demand_trend
)


def run_reporting_pipeline(
    df,
    model,
    y_test,
    predictions,
    feature_names
):

    plot_demand_trend(df)

    plot_actual_vs_forecast(
        y_test,
        predictions
    )

    plot_feature_importance(
        model,
        feature_names
    )

    plot_residuals(
        y_test,
        predictions
    )