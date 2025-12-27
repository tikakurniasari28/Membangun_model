import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
os.environ["MLFLOW_DISABLE_DATABRICKS"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from mlflow.models.signature import infer_signature
import mlflow
import mlflow.sklearn

mlflow.set_experiment("Skilled-Tuning-ManualLogging")

data = pd.read_csv("Membangun_model\day_wise_processed.csv")

y = data["Confirmed"]
X = data.drop(columns=["Confirmed"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

n_estimators_list = [50, 100]
max_depth_list = [5, 10]

best_mse = float("inf")
best_params = None

for n_estimators in n_estimators_list:
    for max_depth in max_depth_list:
        with mlflow.start_run():

            model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            )
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            # Manual logging (params)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            # Manual logging (metrics)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            signature = infer_signature(X_train, model.predict(X_train))

            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5)
            )

            print(
                f"Run selesai | n_estimators={n_estimators}, "
                f"max_depth={max_depth}, MSE={mse:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}"
            )

            if mse < best_mse:
                best_mse = mse
                best_params = {"n_estimators": n_estimators, "max_depth": max_depth}

print("\n Best Params:", best_params)
print(" Best MSE:", best_mse)
