from src.model import (
    prepare_features_and_target,
    train_test_split_time_series,
    train_model,
    make_predictions,
    save_model
)

from src.evaluation import (
    evaluate_model
)


def run_model_pipeline(df):

    X, y = prepare_features_and_target(df)

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split_time_series(
        X,
        y
    )

    model = train_model(
        X_train,
        y_train
    )

    predictions = make_predictions(
        model,
        X_test
    )

    metrics = evaluate_model(
        y_test,
        predictions
    )

    save_model(
        model,
        "models/xgboost_forecaster.pkl"
    )

    return {
        "model": model,
        "metrics": metrics,
        "predictions": predictions,
        "y_test": y_test,
        "X_test": X_test,
        "feature_names": X.columns
    }