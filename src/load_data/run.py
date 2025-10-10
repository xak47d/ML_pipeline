"""
This script download a URL to a local destination
"""
import argparse
import logging
import os

import wandb

import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)
    

def go(args):
    """
    Carga un conjunto de datos local a Weights & Biases como un artefacto
    """
    logger.info("*"*50)
    logger.info("Iniciando proceso de carga de datos")
    logger.info("*"*50)

    run = wandb.init(
        job_type="load_data"
    )
    run.config.update(args)

    logger.info(f"Cargando {args.artifact_name} a Weights & Biases")

    artifact = wandb.Artifact(
        args.artifact_name,
        type=args.artifact_type,
        description=args.artifact_description,
    )

    artifact.add_file(os.path.join("data", args.dataset))
    
    run.log_artifact(artifact)

    artifact.wait()

    logger.info("*"*50)
    logger.info("Proceso de carga de datos finalizado")
    logger.info("*"*50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Carga un conjunto de datos local a Weights & Biases como un artefacto"
        )

    parser.add_argument(
            "--dataset", 
            type=str, 
            help="Nombre del conjunto de datos a cargar (debe estar en la carpeta data)"
        )

    parser.add_argument(
            "--artifact_name", 
            type=str, 
            help="Nombre del artefacto de salida"
        )

    parser.add_argument(
        "--artifact_type", 
        type=str, 
        help="Tipo de artefacto de salida"
    )

    parser.add_argument(
        "--artifact_description", 
        type=str, 
        help="Una breve descripción de este artefacto"
    )

    args = parser.parse_args()

    go(args)