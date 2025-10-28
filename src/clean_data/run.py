#!/usr/bin/env python
"""
Realiza operaciones básicas de limpieza en el conjunto de datos y guarda los
resultados en Weights & Biases.
"""
import os
import argparse
import logging
import pandas as pd
import numpy as np

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
    
    # Definir un mapa de limpieza centralizado
    # Agrupar reemplazos de valores por columna
    cleaning_map = {
        'Performance': {'gOOD': 'Good', 'aVERAGE': 'Average', 'vG': 'Vg', 'eXCELLENT': 'Excellent'},
        'Gender': {'MALE': 'male', 'FEMALE': 'female'},
        'Caste': {'st': 'ST', 'obc': 'OBC', 'gENERAL': 'General', 'sc': 'SC'},
        'coaching': {'wa': 'WA', 'no': 'NO', 'oa': 'OA'},
        'time': {'one': 'ONE', 'two': 'TWO', 'three': 'THREE'},
        'Class_ten_education': {'seba': 'SEBA', 'cbse': 'CBSE', 'others': 'OTHERS'},
        'twelve_education': {'cbse': 'CBSE', 'ahsec': 'AHSEC'},
        'medium': {'english': 'ENGLISH', 'others': 'OTHERS', 'assamese': 'ASSAMESE'},
        'Class_ X_Percentage': {'eXCELLENT': 'Excellent', 'vG': 'Vg', 'gOOD': 'Good'},
        'Class_XII_Percentage': {'eXCELLENT': 'Excellent', 'vG': 'Vg', 'gOOD': 'Good'},
        'Father_occupation': {
            'business': 'BUSINESS', 'bank_official': 'BANK_OFFICIAL',
            'college_teacher': 'COLLEGE_TEACHER', 'others': 'OTHERS',
            'school_teacher': 'SCHOOL_TEACHER', 'cultivator': 'CULTIVATOR',
            'engineer': 'ENGINEER', 'doctor': 'DOCTOR'
        },
        'Mother_occupation': {
            'school_teacher': 'SCHOOL_TEACHER', 'doctor': 'DOCTOR',
            'house_wife': 'HOUSE_WIFE', 'others': 'OTHERS'
        }
    }
    
    # Aplicar limpieza en un solo bucle
    for col, replacements in cleaning_map.items():
        # Encadenar .str.strip() y .replace()
        df[col] = df[col].str.strip().replace(replacements)
        
    # Reemplazar valores nulos genéricos
    df.replace(['', ' ', 'NAN'], np.nan, inplace=True)
    
    # Obtener la lista de columnas que acabamos de limpiar
    cols_to_impute = [col for col in cleaning_map.keys() if col in df.columns]
    
    # Calcular todas las modas necesarias de una vez
    # Usar .mode().loc[0] para obtener el primer valor (la moda)
    imputation_values = {col: df[col].mode().loc[0] for col in cols_to_impute}
    
    # Aplicar fillna UNA SOLA VEZ con el diccionario de modas
    df.fillna(imputation_values, inplace=True)
    
    # Eliminar columna no deseada
    col_to_drop = 'mixed_type_col'
    df.drop(col_to_drop, axis=1, inplace=True)
    
    # Guardar y registrar el artefacto de salida (sin cambios)
    filename = args.output_artifact
    df.to_csv(filename, index=False)
    
    logger.info(f"Forma del dataframe resultante: {df.shape}")
    
    artifact = wandb.Artifact(
        filename,
        type=args.output_type,
        description=args.output_description,
    )
    artifact.add_file(filename)
    
    run.log_artifact(artifact)
    artifact.wait()
    
    # Limpieza local
    os.remove(filename)

    logger.info("*" * 50)
    logger.info("Proceso de limpieza finalizado")
    logger.info("*" * 50)
    
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