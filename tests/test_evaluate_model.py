import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

from src.test_model.evaluate_model_for_test import evaluate_model

@pytest.fixture
def evaluation_data_fixture():
    """
    Preparar los datos falsos para la evaluación.
    
    Usar datos codificados (números en lugar de 'Good', 'Average', 'Vg').
    
    Asumamos una codificación:
    'Good'    -> 1
    'Average' -> 0
    'Vg'      -> 2
    """
    # Valores verdaderos (codificados)
    # Original: ['Good', 'Average', 'Good', 'Vg', 'Average']
    y_test_truth_encoded = pd.Series([1, 0, 1, 2, 0])
    
    # Valores Falsos (codificados)
    # Original: ['Good', 'Average', 'Vg', 'Vg', 'Good']
    y_pred_fake_encoded = pd.Series([1, 0, 2, 2, 1])
    
    expected_metrics = {
        'accuracy': 0.6,
        'f1_macro': 0.611111111111111,
        'f1_weighted': 0.6,
        'precision_macro': 0.6666666666666666,
        'recall_macro': 0.6666666666666666
    }
    
    return y_test_truth_encoded, y_pred_fake_encoded, expected_metrics

def test_calculate_correct_metrics(evaluation_data_fixture):
    """
    Prueba que la función 'test_model' calcula
    correctamente todas las métricas.
    """
    
    y_test, y_pred, expected_metrics = evaluation_data_fixture
    
    # Llamar directamente a la función de cálculo
    calculated_metrics = evaluate_model(y_test, y_pred)
    
    # Comparar los resultados
    assert calculated_metrics['accuracy'] == pytest.approx(expected_metrics['accuracy'])
    assert calculated_metrics['f1_macro'] == pytest.approx(expected_metrics['f1_macro'])
    assert calculated_metrics['f1_weighted'] == pytest.approx(expected_metrics['f1_weighted'])
    assert calculated_metrics['precision_macro'] == pytest.approx(expected_metrics['precision_macro'])
    assert calculated_metrics['recall_macro'] == pytest.approx(expected_metrics['recall_macro'])