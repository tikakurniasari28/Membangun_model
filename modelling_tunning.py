import os
os.environ["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] = "false"
os.environ["MLFLOW_DISABLE_DATABRICKS"] = "true"
os.environ["MLFLOW_TRACKING_URI"] = "file:./mlruns"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn


data = pd.read_csv("Membangun_model/day_wise_processed.csv")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


n_estimators_list = [50, 100]
max_depth_list = [5, 10]

best_mse = float("inf")

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
            r2 = r2_score(y_test, y_pred)

            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("max_depth", max_depth)

            mlflow.log_metric("mse", mse)
            mlflow.log_metric("r2", r2)

            mlflow.sklearn.log_model(model, "model")

            print(
                f"Run selesai | n_estimators={n_estimators}, "
                f"max_depth={max_depth}, MSE={mse}, R2={r2}"
            )

            if mse < best_mse:
                best_mse = mse
