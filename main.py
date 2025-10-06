#!/usr/bin/env python
"""
MLFlow Pipeline to train and test a Random Forest
"""
import json
import os
import hydra
import mlflow
from omegaconf import DictConfig


STEPS = [
    "load_data",
    "clean_data",
    "train_val_test_split_data",
    
]


@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):
    """Main pipeline function"""
    
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else STEPS

    if "load_data" in active_steps:
        mlflow.run(
            "./src/load_data",
            "main",
            parameters={
                "sample": config["data_processing"]["sample"],
                "artifact_name": "student_entry_performance_modified.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Raw file as downloaded"
            },
        )

    if "eda" in active_steps:
        mlflow.run(
            "./src/eda",
            "main",
        )

    if "clean_data" in active_steps:
        mlflow.run(
            "./src/clean_data",
            "main",
            parameters={
                "input_artifact": "student_entry_performance_modified.csv:latest",
                "output_artifact": "clean_data.csv",
                "output_type": "cleaned_data",
                "output_description": "Data cleaned"
            },
        )

    if "train_val_test_split_data" in active_steps:
        mlflow.run(
            "./src/train_val_test_split_data",
            "main",
            parameters={
                "input_artifact": "clean_data.csv:latest",
                "test_size": config["data_processing"]["test_size"],
                "val_size": config["data_processing"]["val_size"],
                "random_state": config["data_processing"]["random_state"],
                "stratify_column": config["data_processing"]["stratify_column"]
            },
        )


if __name__ == "__main__":
    go()