#!/usr/bin/env python
"""
Performs basic cleaning on the data and store the results in Weights & Biases
"""
import os
import argparse
import logging
import tempfile
import pandas as pd
from sklearn.model_selection import train_test_split

import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Separación de datos en conjuntos de train_val/test
    """
    logger.info("*"*50)
    logger.info("Iniciando proceso de segregación de datos")
    logger.info("*"*50)

    run = wandb.init(
            job_type="segregate"
        )
    run.config.update(args)

    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    logger.info(f"Datos cargados: {df.shape}")

    stratify = None
    if args.stratify_column != "none" and args.stratify_column in df.columns:
        stratify = df[args.stratify_column]
        logger.info(f"Stratificando por columna: {args.stratify_column}")

    train_val, test = train_test_split(
        df,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=stratify
    )

    logger.info(f"Tamaños de los splits:")
    logger.info(f"  Train: {len(train_val)} ({len(train_val)/len(df)*100:.1f}%)")
    logger.info(f"  Test:  {len(test)} ({len(test)/len(df)*100:.1f}%)")

    for dataframe, k in zip([train_val, test], ["train_val", "test"]):
        logger.info(f"Subiendo {k}_data.csv a W&B")
        with tempfile.NamedTemporaryFile("w") as fp:

            dataframe.to_csv(fp.name, index=False)

            artifact = wandb.Artifact(
                name=f"{k}_data.csv",
                type=f"{k}_data",
                description=f"{k} data split",
            )

            artifact.add_file(fp.name)
            run.log_artifact(artifact)

            artifact.wait()



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Segrega los datos en conjuntos de train_val/test"
        )
    
    parser.add_argument(
            "--input_artifact",
            type=str,
            help="Archivo de entrada",
        )

    parser.add_argument(
            "--test_size",
            type=float,
            help="Proporción del conjunto de prueba"
        )

    parser.add_argument(
            "--random_state",
            type=int,
            help="Semilla para la reproducibilidad"
        )

    parser.add_argument(
            "--stratify_column",
            type=str,
            help="Columna para la estratificación"
        )

    args = parser.parse_args()
    
    go(args)