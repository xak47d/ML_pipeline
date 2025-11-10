import pandas as pd
import logging

from sklearn.metrics import (
    accuracy_score, 
    f1_score, 
    classification_report, 
    precision_score, 
    recall_score     
)

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def evaluate_model(y_true_encoded, y_pred):
    """
    Calcula y devuelve un diccionario con las métricas de evaluación.
    """
    logger.info("Calculando métricas...")
    metrics = {
        'accuracy': accuracy_score(y_true_encoded, y_pred),
        'f1_macro': f1_score(y_true_encoded, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true_encoded, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true_encoded, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true_encoded, y_pred, average='macro', zero_division=0)
    }
    return metrics