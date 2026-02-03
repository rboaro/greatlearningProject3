# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Prepares raw data and provides training and test datasets.
"""
print("DATA_PREP VERSION: Segment encoding v1")

import os
import argparse
import logging
import mlflow
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to input data (CSV file)")
    parser.add_argument("--test_train_ratio", type=float, default=0.2)
    parser.add_argument("--train_data", type=str, required=True, help="Folder path to save train data")
    parser.add_argument("--test_data", type=str, required=True, help="Folder path to save test data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Start MLflow Run
    mlflow.start_run()

    # Log arguments
    logging.info(f"Input data path: {args.data}")
    logging.info(f"Test-train ratio: {args.test_train_ratio}")
    logging.info(f"Train output folder: {args.train_data}")
    logging.info(f"Test output folder: {args.test_data}")

    # Reading Data
    df = pd.read_csv(args.data)

    # Encode categorical feature
    le = LabelEncoder()
    df["transmission_encoded"] = le.fit_transform(df["transmission"])

    # Split Data into train and test datasets
    train_df, test_df = train_test_split(df, test_size=args.test_train_ratio, random_state=42)

    # Save train and test data
    os.makedirs(args.train_data, exist_ok=True)
    os.makedirs(args.test_data, exist_ok=True)

    train_df.to_csv(os.path.join(args.train_data, "used_cars.csv"), index=False)
    test_df.to_csv(os.path.join(args.test_data, "used_cars.csv"), index=False)

    # log the metrics
    mlflow.log_metric("train size", train_df.shape[0])
    mlflow.log_metric("test size", test_df.shape[0])

    mlflow.end_run()

if __name__ == "__main__":
    main()