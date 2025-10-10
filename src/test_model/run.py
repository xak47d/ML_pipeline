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
    """
    Componente de prueba del modelo entrenado
    """
    logger.info("*"*50)
    logger.info("Iniciando proceso de prueba del modelo")
    logger.info("*"*50)

    run = wandb.init(
        project="my_ml_project",
        job_type="test_model"
    )

    run.config.update(args)

    logger.info("Descargando artefactos")

    model_local_path = run.use_artifact(args.mlflow_model).download()

    test_dataset_path = run.use_artifact(args.test_dataset).file()

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("Performance")

    logger.info("Cargando modelo y realizando inferencia en el conjunto de prueba")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Evaluando el modelo")

    test_accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy de prueba: {test_accuracy}")

    run.summary['test_accuracy'] = test_accuracy

    logger.info("*"*50)
    logger.info("Proceso de Prueba finalizado")
    logger.info("*"*50)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Probar un modelo MLflow con un conjunto de datos de prueba"
        )

    parser.add_argument(
        "--mlflow_model",
        type=str, 
        help="Modelo MLflow a probar",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str, 
        help="Conjunto de datos de prueba",
        required=True
    )

    args = parser.parse_args()

    go(args)