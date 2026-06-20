import matplotlib.pyplot as plt
from pathlib import Path


FIGURES_DIR = Path(
    "reports/figures"
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def plot_actual_vs_forecast(
    y_true,
    predictions
):

    plt.figure(
        figsize=(15, 6)
    )

    plt.plot(
        y_true.values,
        label="Actual"
    )

    plt.plot(
        predictions,
        label="Forecast"
    )

    plt.title(
        "Actual vs Forecast Demand"
    )

    plt.xlabel("Time")

    plt.ylabel(
        "Demand (MW)"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "actual_vs_forecast.png"
    )

    plt.close()
def plot_feature_importance(
    model,
    feature_names
):

    importance = (
        model.feature_importances_
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.barh(
        feature_names,
        importance
    )

    plt.title(
        "Feature Importance"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "feature_importance.png"
    )

    plt.close()
def plot_residuals(
    y_true,
    predictions
):

    residuals = (
        y_true -
        predictions
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.scatter(
        predictions,
        residuals
    )

    plt.axhline(
        y=0
    )

    plt.title(
        "Residual Analysis"
    )

    plt.xlabel(
        "Predicted Demand"
    )

    plt.ylabel(
        "Residual Error"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "residuals.png"
    )

    plt.close()
def plot_demand_trend(
    df
):

    plt.figure(
        figsize=(15, 6)
    )

    plt.plot(
        df["Settlement Date"],
        df["Scheduled Demand (MW)"]
    )

    plt.title(
        "Electricity Demand Trend"
    )

    plt.xlabel(
        "Date"
    )

    plt.ylabel(
        "Demand (MW)"
    )

    plt.tight_layout()

    plt.savefig(
        FIGURES_DIR /
        "demand_trend.png"
    )

    plt.close()