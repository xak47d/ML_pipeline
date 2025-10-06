"""
This script download a URL to a local destination
"""
import argparse
import logging
import os

import wandb

import logging

import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)
    

def go(args):

    run = wandb.init(
        project="my_ml_project",
        job_type="load_data"
    )

    run.config.update(args)

    logger.info("Downloading artifacts")

    model_local_path = run.use_artifact(args.mlflow_model).download()

    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Scoring")

    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy}")

    run.summary['accuracy'] = accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download URL to a local destination")

    parser.add_argument("sample", type=str, help="Name of the sample to download")

    parser.add_argument("artifact_name", type=str, help="Name for the output artifact")

    parser.add_argument("artifact_type", type=str, help="Output artifact type.")

    parser.add_argument(
        "artifact_description", type=str, help="A brief description of this artifact"
    )

    args = parser.parse_args()

    go(args)