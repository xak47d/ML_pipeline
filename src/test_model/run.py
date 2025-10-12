"""
This script download a URL to a local destination
"""
import argparse
import logging
import os

import wandb
import pickle

import logging

import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, classification_report

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
        job_type="test_model"
    )

    run.config.update(args)

    logger.info("Descargando artefactos")

    model_local_path = run.use_artifact(args.mlflow_model).download()

    test_dataset_path = run.use_artifact(args.test_dataset).file()

    label_encoder_path = os.path.join(model_local_path, "label_mapping.pkl")

    with open(label_encoder_path, "rb") as f:
        label_encoder = pickle.load(f)

    # Read test dataset
    X_test = pd.read_csv(test_dataset_path)
    y_test = X_test.pop("Performance")

    y_test_encoded = label_encoder.transform(y_test)

    logger.info("Cargando modelo y realizando inferencia en el conjunto de prueba")
    sk_pipe = mlflow.sklearn.load_model(model_local_path)
    y_pred = sk_pipe.predict(X_test)

    logger.info("Evaluando el modelo")

    test_accuracy = accuracy_score(y_test_encoded, y_pred)
    test_f1_macro = f1_score(y_test_encoded, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test_encoded, y_pred, average='weighted')

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test F1 Macro: {test_f1_macro:.4f}")
    logger.info(f"Test F1 Weighted: {test_f1_weighted:.4f}")

    y_pred_labels = label_encoder.inverse_transform(y_pred)
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(y_test, y_pred_labels))

    run.summary['test_accuracy'] = test_accuracy
    run.summary['test_f1_macro'] = test_f1_macro
    run.summary['test_f1_weighted'] = test_f1_weighted

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