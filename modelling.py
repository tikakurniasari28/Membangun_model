import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
data_path = BASE_DIR / "day_wise_processed.csv"

def main():
    # Tracking lokal
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Basic-Autolog")

    df = pd.read_csv(data_path)

 
    y = df["Confirmed"]
    X = df.drop(columns=["Confirmed"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    mlflow.sklearn.autolog()

    with mlflow.start_run():
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Training selesai")
        print("MSE:", mse)
        print("R2:", r2)

if __name__ == "__main__":
    main()