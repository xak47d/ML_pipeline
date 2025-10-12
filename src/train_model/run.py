"""
This script download a URL to a local destination
"""
import argparse
import logging
import os
import shutil
import json

import wandb
import pickle

import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from xgboost import XGBClassifier

from feature_engineering_pipeline import get_preprocessor

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)
    

def go(args):
    """
    Función de entrenamiento de un modelo XGBoost
    """
    logger.info("*"*50)
    logger.info("Iniciando proceso de entrenamiento")
    logger.info("*"*50)

    run = wandb.init(
            job_type="train_xgboost"
        )
    run.config.update(args)

    # Leer la configuración del modelo
    with open(args.model_config) as fp:
        xgb_config = json.load(fp)
    run.config.update(xgb_config)

    # Fija la semilla para reproducibilidad
    xgb_config['random_state'] = args.random_seed

    trainval_local_path = run.use_artifact(args.train_artifact).file()

    X = pd.read_csv(trainval_local_path)
    y = X.pop("Performance")

    logger.info(f"Datos: {X.shape[0]} muestras, {X.shape[1]} features")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    class_mapping = dict(
            zip(
                label_encoder.classes_, 
                label_encoder.transform(label_encoder.classes_)
            )
        )

    logger.info(f"Mapeo de clases: {class_mapping}")

    run.config.update({"class_mapping": class_mapping})

    preprocessor = get_preprocessor()

    xgb_model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss', 
        **xgb_config
    )

    model_pipeline = Pipeline(steps=[
            ('preprocesador', preprocessor), 
            ('modelo', xgb_model)
        ])
    
    skfold = StratifiedKFold(
            n_splits=5, 
            shuffle=True, 
            random_state=args.random_seed
        )
    cv_scores = cross_val_score(
            model_pipeline, 
            X, 
            y_encoded, 
            cv=skfold, 
            scoring='accuracy', 
            n_jobs=-1
        )
    
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()

    logger.info(f"CV Scores: {cv_scores}")
    logger.info(f"CV Accuracy: {cv_mean:.4f} (+/- {cv_std:.4f})")
    
    run.summary['cv_accuracy_mean'] = cv_mean
    run.summary['cv_accuracy_std'] = cv_std

    for i, score in enumerate(cv_scores):
        run.summary[f'cv_fold_{i+1}_accuracy'] = score

    results = {
        'cv_accuracy_mean': float(cv_mean),
        'cv_accuracy_std': float(cv_std),
        'cv_scores': [float(s) for s in cv_scores],
        'model_config': xgb_config
    }

    with open("train_results.json", 'w') as f:
        json.dump(results, f, indent=2)

    logger.info("Entrenando el modelo")

    model_pipeline.fit(X, y_encoded)
    y_pred = model_pipeline.predict(X)
    
    train_accuracy = accuracy_score(y_encoded, y_pred)
    train_f1_macro = f1_score(y_encoded, y_pred, average='macro')
    train_f1_weighted = f1_score(y_encoded, y_pred, average='weighted')

    logger.info(f"Train Accuracy: {train_accuracy:.4f}")
    logger.info(f"F1 Macro: {train_f1_macro:.4f}")
    
    run.summary['train_accuracy'] = train_accuracy
    run.summary['train_f1_macro'] = train_f1_macro
    run.summary['train_f1_weighted'] = train_f1_weighted

    with open("label_mapping.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    # Guardamos el modelo en formato mlflow
    if os.path.exists("xgboost_dir"):
        shutil.rmtree("xgboost_dir")

    signature = mlflow.models.infer_signature(X, y_pred)

    mlflow.sklearn.save_model(
        model_pipeline,
        path="xgboost_dir",
        serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
        signature=signature,
        input_example=X.iloc[:5],
    )

    logger.info('Creando y subiendo el artefacto del modelo')
    artifact = wandb.Artifact(
        args.output_artifact,
        type="model_export",
        description="Xgboost pipeline export",
        metadata={
            'cv_accuracy_mean': cv_mean,
            'train_accuracy': train_accuracy,
            'model_config': xgb_config
        }
    )

    artifact.add_dir(local_path='xgboost_dir')
    artifact.add_file("label_mapping.pkl")

    run.log_artifact(artifact)

    logger.info("*"*50)
    logger.info("Finalizado")
    logger.info("*"*50)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(
            description="Componente de entrenamiento del modelo"
        )

    parser.add_argument(
            "--train_artifact",
            type=str,
            help="Dataset usado para entrenar el modelo"
        )

    parser.add_argument(
            "--model_config",
            help="Configuración del modelo. Un path a un archivo JSON con la configuración que se pasará al constructor del modelo.",
            default="{}",
        )

    parser.add_argument(
            "--output_artifact",
            type=str,
            help="Nombre del artefacto de salida",
            required=True,
        )

    parser.add_argument(
            "--random_seed",
            type=int,
            help="Semilla para la reproducibilidad",
            default=42,
            required=False,
        )

    parser.add_argument(
            "--stratify_by",
            type=str,
            help="Columna para estratificar el conjunto de datos",
            default="none",
            required=False,
        )
 
    args = parser.parse_args()

    go(args)