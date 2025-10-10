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

    df['Performance'] = df['Performance'].str.strip()
    df['Performance'] = df['Performance'].replace('gOOD', 'Good')
    df['Performance'] = df['Performance'].replace('aVERAGE', 'Average')
    df['Performance'] = df['Performance'].replace('vG', 'Vg')
    df['Performance'] = df['Performance'].replace('eXCELLENT', 'Excellent')

    df['Gender'] = df['Gender'].str.strip()
    df['Gender'] = df['Gender'].replace('MALE', 'male')
    df['Gender'] = df['Gender'].replace('FEMALE', 'female')

    df['Caste'] = df['Caste'].str.strip()
    df['Caste'] = df['Caste'].replace('st', 'ST')
    df['Caste'] = df['Caste'].replace('obc', 'OBC')
    df['Caste'] = df['Caste'].replace('gENERAL', 'General')
    df['Caste'] = df['Caste'].replace('sc', 'SC')

    df['coaching'] = df['coaching'].str.strip()
    df['coaching'] = df['coaching'].replace('wa', 'WA')
    df['coaching'] = df['coaching'].replace('no', 'NO')
    df['coaching'] = df['coaching'].replace('oa', 'OA')

    df['time'] = df['time'].str.strip()
    df['time'] = df['time'].replace('one', 'ONE')
    df['time'] = df['time'].replace('two', 'TWO')
    df['time'] = df['time'].replace('three', 'THREE')

    df['Class_ten_education'] = df['Class_ten_education'].str.strip()
    df['Class_ten_education'] = df['Class_ten_education'].replace('seba', 'SEBA')
    df['Class_ten_education'] = df['Class_ten_education'].replace('cbse', 'CBSE')
    df['Class_ten_education'] = df['Class_ten_education'].replace('others', 'OTHERS')

    df['twelve_education'] = df['twelve_education'].str.strip()
    df['twelve_education'] = df['twelve_education'].replace('cbse', 'CBSE')
    df['twelve_education'] = df['twelve_education'].replace('ahsec', 'AHSEC')

    df['medium'] = df['medium'].str.strip()
    df['medium'] = df['medium'].replace('english', 'ENGLISH')
    df['medium'] = df['medium'].replace('others', 'OTHERS')
    df['medium'] = df['medium'].replace('assamese', 'ASSAMESE')

    df['Class_ X_Percentage'] = df['Class_ X_Percentage'].str.strip()
    df['Class_ X_Percentage'] = df['Class_ X_Percentage'].replace('eXCELLENT', 'Excellent')
    df['Class_ X_Percentage'] = df['Class_ X_Percentage'].replace('vG', 'Vg')
    df['Class_ X_Percentage'] = df['Class_ X_Percentage'].replace('gOOD', 'Good')

    df['Class_XII_Percentage'] = df['Class_XII_Percentage'].str.strip()
    df['Class_XII_Percentage'] = df['Class_XII_Percentage'].replace('eXCELLENT', 'Excellent')
    df['Class_XII_Percentage'] = df['Class_XII_Percentage'].replace('vG', 'Vg')
    df['Class_XII_Percentage'] = df['Class_XII_Percentage'].replace('gOOD', 'Good')

    df['Father_occupation'] = df['Father_occupation'].str.strip()
    df['Father_occupation'] = df['Father_occupation'].replace('business', 'BUSINESS')
    df['Father_occupation'] = df['Father_occupation'].replace('bank_official', 'BANK_OFFICIAL')
    df['Father_occupation'] = df['Father_occupation'].replace('college_teacher', 'COLLEGE_TEACHER')
    df['Father_occupation'] = df['Father_occupation'].replace('others', 'OTHERS')
    df['Father_occupation'] = df['Father_occupation'].replace('school_teacher', 'SCHOOL_TEACHER')
    df['Father_occupation'] = df['Father_occupation'].replace('cultivator', 'CULTIVATOR')
    df['Father_occupation'] = df['Father_occupation'].replace('engineer', 'ENGINEER')
    df['Father_occupation'] = df['Father_occupation'].replace('doctor', 'DOCTOR')

    df['Mother_occupation'] = df['Mother_occupation'].str.strip()
    df['Mother_occupation'] = df['Mother_occupation'].replace('school_teacher', 'SCHOOL_TEACHER')
    df['Mother_occupation'] = df['Mother_occupation'].replace('doctor', 'DOCTOR')
    df['Mother_occupation'] = df['Mother_occupation'].replace('house_wife', 'HOUSE_WIFE')
    df['Mother_occupation'] = df['Mother_occupation'].replace('others', 'OTHERS')

    performance_mode = df['Performance'].mode()[0]
    df.replace(['', ' ', 'NAN'], np.nan, inplace=True)
    df.fillna({'Performance': performance_mode}, inplace=True)

    gender_mode = df['Gender'].mode()[0]
    df.fillna({'Gender': gender_mode}, inplace=True)

    caste_mode = df['Caste'].mode()[0]
    df.fillna({'Caste': caste_mode}, inplace=True)

    coaching_mode = df['coaching'].mode()[0]
    df.fillna({'coaching': coaching_mode}, inplace=True)

    time_mode = df['time'].mode()[0]
    df.fillna({'time': time_mode}, inplace=True)

    class_ten_education_mode = df['Class_ten_education'].mode()[0]
    df.fillna({'Class_ten_education': class_ten_education_mode}, inplace=True)

    twelve_education_mode = df['twelve_education'].mode()[0]
    df.fillna({'twelve_education': twelve_education_mode}, inplace=True)

    medium_mode = df['medium'].mode()[0]
    df.fillna({'medium': medium_mode}, inplace=True)

    class_x_percentage_mode = df['Class_ X_Percentage'].mode()[0]
    df.fillna({'Class_ X_Percentage': class_x_percentage_mode}, inplace=True)

    class_xii_percentage_mode = df['Class_XII_Percentage'].mode()[0]
    df.fillna({'Class_XII_Percentage': class_xii_percentage_mode}, inplace=True)

    father_occupation_mode = df['Father_occupation'].mode()[0]
    df.fillna({'Father_occupation': father_occupation_mode}, inplace=True)

    mother_occupation_mode = df['Mother_occupation'].mode()[0]
    df.fillna({'Mother_occupation': mother_occupation_mode}, inplace=True)

    df.drop('mixed_type_col', axis=1, inplace=True)

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