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
from typing import Any, Dict, List, Optional
import pandas as pd

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

# lazy model loader
_MODEL: Optional[Any] = None

def _model_try_paths() -> List[str]:
    """Return candidate paths to look for model artifacts without calling hydra at import time."""
    paths = [
        os.path.join(os.getcwd(), "xgboost_dir"),
        "xgboost_dir",
    ]
    try:
        # call get_original_cwd only when hydra is available/initialized
        orig = hydra.utils.get_original_cwd()
        if orig:
            paths.insert(1, os.path.join(orig, "xgboost_dir"))
    except Exception:
        # hydra not initialized yet — ignore
        pass
    return paths


def _load_model():
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    for p in _model_try_paths():
        if not p:
            continue
        if os.path.exists(p):
            try:
                _MODEL = mlflow.pyfunc.load_model(p)
                logger.info(f"Loaded model via pyfunc from {p}")
                return _MODEL
            except Exception:
                try:
                    _MODEL = mlflow.sklearn.load_model(p)
                    logger.info(f"Loaded sklearn model from {p}")
                    return _MODEL
                except Exception as e:
                    logger.warning(f"Failed loading model from {p}: {e}")
    raise RuntimeError("Model not found in expected locations: " + ", ".join(_model_try_paths()))
# ...existing code...

@app.get("/info", response_model=None)
def info():
    """Service info and model metadata (non-sensitive)."""
    info_payload = {
        "service": "ml-pipeline",
        "version": "unknown",
        "endpoints": ["/info", "/health", "/ready", "/model_execution", "/batch_model_execution", "/docs"],
    }
    try:
        m = _load_model()
        info_payload.update({"model_loaded": True, "model_type": type(m).__name__})
    except Exception as e:
        info_payload.update({"model_loaded": False, "model_error": str(e)})
    return info_payload


@app.get("/health", response_model=None)
def health():
    """Liveness probe."""
    return {"status": "alive"}


@app.get("/ready", response_model=None)
def ready():
    """Readiness probe: true if model can be loaded."""
    try:
        _load_model()
        return {"ready": True}
    except Exception as e:
        return {"ready": False, "reason": str(e)}


def _normalize_instances(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and "instances" in payload:
        return payload["instances"]
    if isinstance(payload, dict) and "instance" in payload:
        return [payload["instance"]]
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    raise ValueError("Invalid payload format. Send an object, list, or {'instances': [...]}.")


@app.post("/model_execution", response_model=None)
def model_execution(payload: Dict[str, Any]):
    """Single-record inference (accepts object or {'instance': {...}})."""
    try:
        instances = _normalize_instances(payload)
        if len(instances) > 1:
            return {"error": "Use /batch_model_execution for multiple records."}
        df = pd.DataFrame(instances)
        model = _load_model()
        preds = model.predict(df)
        try:
            preds_out = preds.tolist()
        except Exception:
            preds_out = list(preds)
        probs_out = None
        if hasattr(model, "predict_proba"):
            try:
                probs_out = model.predict_proba(df).tolist()
            except Exception:
                probs_out = None
        return {"predictions": preds_out, "probabilities": probs_out}
    except Exception as e:
        logger.exception("model_execution failed")
        return {"error": str(e)}


@app.post("/batch_model_execution", response_model=None)
def batch_model_execution(payload: Any):
    """Batch inference. Accepts list of records or {'instances': [...]}."""
    try:
        instances = _normalize_instances(payload)
        df = pd.DataFrame(instances)
        model = _load_model()
        preds = model.predict(df)
        try:
            preds_out = preds.tolist()
        except Exception:
            preds_out = list(preds)
        probs_out = None
        if hasattr(model, "predict_proba"):
            try:
                probs_out = model.predict_proba(df).tolist()
            except Exception:
                probs_out = None
        return {"predictions": preds_out, "probabilities": probs_out}
    except Exception as e:
        logger.exception("batch_model_execution failed")
        return {"error": str(e)}
