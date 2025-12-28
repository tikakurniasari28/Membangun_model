import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "day_wise_processed.csv"

def main():
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Skilled-Manual-Tuning")

    df = pd.read_csv(data_path)

    y = df["Confirmed"]
    X = df.drop(columns=["Confirmed"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    input_example = X_train.iloc[:5]

    fit_intercept_range = [True, False]
    positive_range = [True, False]

    best_mse = float("inf")
    best_params = None

    for fit_intercept in fit_intercept_range:
        for positive in positive_range:
            with mlflow.start_run(run_name=f"lr_fit_{fit_intercept}_pos_{positive}"):
                mlflow.set_tag("model_name", "LinearRegression")
                mlflow.set_tag("run_type", "tuning_manual_logging")

                mlflow.log_param("fit_intercept", fit_intercept)
                mlflow.log_param("positive", positive)

                model = LinearRegression(
                    fit_intercept=fit_intercept,
                    positive=positive
                )

                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)

                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)

                mlflow.log_metric("mse", mse)
                mlflow.log_metric("r2", r2)

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path="model",
                    input_example=input_example
                )

                if mse < best_mse:
                    best_mse = mse
                    best_params = {"fit_intercept": fit_intercept, "positive": positive}

    print("Tuning selesai")
    print("Best Params:", best_params)
    print("Best MSE:", best_mse)

if __name__ == "__main__":
    main()
