# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
"""
Registers the best-trained ML model from the sweep job.
"""

import argparse
import os
import mlflow

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to MLflow model folder from previous step")
    parser.add_argument("--model_name", type=str, default="price_prediction_model", help="Registered model name")
    args = parser.parse_args()

    # Start a run so we have a run_id and an artifacts location
    with mlflow.start_run() as run:
        # Log the entire MLflow model directory as an artifact named 'model'
        # (this copies MLmodel + env files + data into MLflow artifacts storage)
        mlflow.log_artifacts(args.model, artifact_path="model")

        model_uri = f"runs:/{run.info.run_id}/model"
        mv = mlflow.register_model(model_uri=model_uri, name=args.model_name)

        print(f"Registered model: name={mv.name}, version={mv.version}", flush=True)

if __name__ == "__main__":
    
    mlflow.start_run()
    
    # Parse Arguments
    args = parse_args()
    
    lines = [
        f"Model name: {args.model_name}",
        f"Model path: {args.model_path}",
        f"Model info output path: {args.model_info_output_path}"
    ]

    for line in lines:
        print(line)

    main(args)

    mlflow.end_run()