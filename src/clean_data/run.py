#!/usr/bin/env python
"""
Realiza operaciones básicas de limpieza en el conjunto de datos y guarda los
resultados en Weights & Biases.
"""
import os
import argparse
import logging
import pandas as pd

import wandb


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)


def go(args):
    """
    Implementación de funciones básicas de limpieza
    """
    logger.info("*"*50)
    logger.info("Iniciando proceso de limpieza")
    logger.info("*"*50)

    run = wandb.init(
        job_type="basic_cleaning"
    )
    run.config.update(args)

    artifact = run.use_artifact(args.input_artifact)
    artifact_path = artifact.file()
    df = pd.read_csv(artifact_path)

    ############################################################################
    # TODO: Añadir aquí las operaciones de limpieza
    ############################################################################

    logger.info(f"Forma del dataframe resultante: {df.shape}")

    filename = args.output_artifact
    df.to_csv(filename, index=False)

    artifact = wandb.Artifact(
        filename,
        type=args.output_type,
        description=args.output_description,
    )

    artifact.add_file(filename)
    run.log_artifact(artifact)

    artifact.wait()

    os.remove(filename)

    logger.info("*"*50)
    logger.info("Proceso de limpieza finalizado")
    logger.info("*"*50)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Este componente realiza una limpieza básica de datos"
        )

    parser.add_argument(
        "--input_artifact",
        type=str,
        help="Archivo objetivo",
        required=True
    )

    parser.add_argument(
        "--output_artifact",
        type=str,
        help="Archivo de salida",
        required=True
    )

    parser.add_argument(
        "--output_type",
        type=str,
        help="Tipo de salida",
        required=True
    )

    parser.add_argument(
        "--output_description",
        type=str,
        help="Archivo procesado por la eliminación de valores atípicos",
        required=True
    )

    args = parser.parse_args()

    go(args)