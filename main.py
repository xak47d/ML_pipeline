#!/usr/bin/env python
"""
MLFlow Pipeline to train and test a Random Forest
"""
import json
import os
import hydra
import mlflow
import logging
from omegaconf import DictConfig
import fastapi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STEPS = [
    "load_data",
    "clean_data",
    "train_test_split_data",
    "train_model",
    "test_model",
]


def run_pipeline_impl(config: DictConfig):
    """Pipeline implementation (callable from both CLI hydra entrypoint and API)."""
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    steps_par = config['main']['steps']
    active_steps = steps_par.split(",") if steps_par != "all" else STEPS

    if "load_data" in active_steps:
        mlflow.run(
            "./src/load_data",
            "main",
            env_manager="local",
            parameters={
                "dataset": config["data_processing"]["dataset"],
                "artifact_name": "data.csv",
                "artifact_type": "raw_data",
                "artifact_description": "Data sin procesar"
            }
        )

    if "eda" in active_steps:
        mlflow.run(
            "./src/eda",
            "main",
            env_manager="local",
        )

    if "clean_data" in active_steps:
        mlflow.run(
            "./src/clean_data",
            "main",
            env_manager="local",
            parameters={
                "input_artifact": "data.csv:latest",
                "output_artifact": "clean_data.csv",
                "output_type": "cleaned_data",
                "output_description": "Datos limpiados por la aplicación de pasos de limpieza"
            },
        )

    if "train_test_split_data" in active_steps:
        mlflow.run(
            "./src/train_test_split_data",
            "main",
            env_manager="local",
            parameters={
                "input_artifact": "clean_data.csv:latest",
                "test_size": config["data_processing"]["test_size"],
                "random_state": config["data_processing"]["random_state"],
                "stratify_column": config["data_processing"]["stratify_column"]
            },
        )

    cv_accuracy = None

    if "train_model" in active_steps:

        xg_config = os.path.abspath("xg_config.json")
        with open(xg_config, "w+") as fp:
            json.dump(
                dict(
                    config["modeling"]["xgboost"].items()
                ),
                fp
            )

        _ = mlflow.run(
            os.path.join(
                hydra.utils.get_original_cwd(),
                "src",
                "train_model"),
            "main",
            env_manager="local",
            parameters={
                "train_artifact": "train_val_data.csv:latest",
                "model_config": xg_config,
                "output_artifact": "model_export",
                "stratify_by": config["modeling"]["stratify_by"],
                "random_seed": config["modeling"]["random_seed"],
            }   
        )

        try:
            results_file = os.path.join(hydra.utils.get_original_cwd(), "train_results.json")
            if os.path.exists(results_file):
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    cv_accuracy = results.get('cv_accuracy_mean', 0.0)
                logger.info(f"CV Accuracy: {cv_accuracy:.4f}")
            else:
                cv_accuracy = 0.0
        except Exception as e:
            logger.error(f"Error leyendo métrica: {e}")
            cv_accuracy = 0.0

    if "test_model" in active_steps:

            _ = mlflow.run(
                f"./src/test_model",
                "main",
                env_manager="local",
                parameters={
                    "mlflow_model": "model_export:latest",
                    "test_dataset": "test_data.csv:latest",
                },
            )

    return cv_accuracy if cv_accuracy is not None else 0.0


@hydra.main(version_base=None, config_path=".", config_name="config")
def go(config: DictConfig):
    """Hydra CLI entrypoint wrapper."""
    return run_pipeline_impl(config)


if __name__ == "__main__":
    go()


# Implement a FastAPI app to expose the pipeline as an API
app = fastapi.FastAPI()

from omegaconf import OmegaConf

@app.post("/run_pipeline/", response_model=None)

@app.get("/")
def root():
    return {"status": "ok", "endpoints": ["/run_pipeline/","/docs","/redoc"]}

def run_pipeline_endpoint(config: dict):
    """API endpoint to run the ML pipeline. Accepts a plain JSON body (dict)."""
    cfg = OmegaConf.create(config)
    cv_accuracy = run_pipeline_impl(cfg)
    return {"cv_accuracy": cv_accuracy}
