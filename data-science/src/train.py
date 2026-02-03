# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Trains ML model using training dataset and evaluates using test dataset. Saves trained model.
"""

import argparse
import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def select_first_csv(path: str) -> str:
    """Pick the first CSV file from a folder (AzureML uri_folder mount)."""
    if os.path.isfile(path):
        return path
    files = [f for f in os.listdir(path) if f.lower().endswith(".csv")]
    if not files:
        raise FileNotFoundError(f"No .csv files found in: {path}")
    files.sort()
    return os.path.join(path, files[0])


def main():
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, required=True, help="Folder containing train CSV")
    parser.add_argument("--test_data", type=str, required=True, help="Folder containing test CSV")
    parser.add_argument("--model_output", type=str, required=True, help="Output folder to save MLflow model")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=None)
    args = parser.parse_args()

    train_path = select_first_csv(args.train_data)
    test_path = select_first_csv(args.test_data)

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    target_col = "price"
    if target_col not in train_df.columns or target_col not in test_df.columns:
        raise ValueError(
            f"Target column '{target_col}' must exist in both train and test CSVs. "
            f"Train cols: {train_df.columns.tolist()} | Test cols: {test_df.columns.tolist()}"
        )

    y_train = train_df[target_col]
    X_train = train_df.drop(columns=[target_col])
    y_test = test_df[target_col]
    X_test = test_df.drop(columns=[target_col])

    with mlflow.start_run():
        model = RandomForestRegressor(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)

        mlflow.log_param("model", "RandomForestRegressor")
        mlflow.log_param("n_estimators", args.n_estimators)
        mlflow.log_param("max_depth", args.max_depth)
        mlflow.log_metric("MSE", float(mse))

        print(f"Mean Squared Error on test set: {mse:.4f}", flush=True)

        # IMPORTANT: because your component output is type=mlflow_model,
        # save an MLflow model folder directly to args.model_output
        os.makedirs(args.model_output, exist_ok=True)
        mlflow.sklearn.save_model(sk_model=model, path=args.model_output)


if __name__ == "__main__":
    main()
