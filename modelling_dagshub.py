import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import mlflow
import mlflow.sklearn

DAGSHUB_USER = "tikakurniasari28"
DAGSHUB_REPO = "Eksperimen_SML_Ni-Luh-Made-Tika-Kurniasari"

DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
if DAGSHUB_TOKEN is None:
    raise ValueError("DAGSHUB_TOKEN belum diset. Jalankan: setx DAGSHUB_TOKEN \"TOKEN_KAMU\" lalu restart terminal.")

mlflow.set_tracking_uri(f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow")

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_experiment("Advanced-DagsHub-ManualLogging")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "day_wise_processed.csv"

df = pd.read_csv(DATA_PATH)

y = df["Confirmed"]
X = df.drop(columns=["Confirmed"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = [
    {"n_estimators": 50, "max_depth": 5},
    {"n_estimators": 100, "max_depth": 10},
]

best_mse = float("inf")

for params in param_grid:
    with mlflow.start_run(run_name=f"rf_{params['n_estimators']}_{params['max_depth']}"):
        model = RandomForestRegressor(random_state=42, **params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_params(params)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        mlflow.sklearn.log_model(model, "model")

        run_id = mlflow.active_run().info.run_id

        plt.figure()
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.title("Prediction vs Actual")
        pred_plot_path = f"pred_vs_actual_{run_id}.png"
        plt.savefig(pred_plot_path)
        plt.close()
        mlflow.log_artifact(pred_plot_path)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), X.columns[indices], rotation=90)
        plt.title("Feature Importance")
        fi_plot_path = f"feature_importance_{run_id}.png"
        plt.tight_layout()
        plt.savefig(fi_plot_path)
        plt.close()
        mlflow.log_artifact(fi_plot_path)

        metrics_json_path = f"metrics_{run_id}.json"
        with open(metrics_json_path, "w") as f:
            json.dump(
                {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2, "params": params},
                f,
                indent=2
            )
        mlflow.log_artifact(metrics_json_path)

        print(f"Run selesai | Params={params} | MSE={mse:.4f} | R2={r2:.4f}")

        if mse < best_mse:
            best_mse = mse
            print("Best model so far!")

print("\nAdvanced selesai")
