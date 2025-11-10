import pytest
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from src.train_model.train_model_for_test import train_model

@pytest.fixture
def sample_training_data():
    """
    Crea datos falsos para probar el entrenamiento.
    """
    X_train_fake = pd.DataFrame({
        'Gender': ['male', 'female', 'male', 'female', 'male', 'female', 'male', 'female', 'male', 'female'],
        'Caste': ['ST', 'OBC', 'General', 'ST', 'SC', 'OBC', 'General', 'ST', 'SC', 'OBC'],
        'coaching': ['WA', 'NO', 'OA', 'WA', 'NO', 'OA', 'WA', 'NO', 'OA', 'WA'],
        'time': ['ONE', 'ONE', 'ONE', 'ONE', 'ONE', 'TWO', 'TWO', 'TWO', 'TWO', 'TWO'],
        'Class_ X_Percentage': ['Excellent', 'Vg', 'Excellent', 'Vg', 'Excellent', 'Vg', 'Excellent', 'Vg', 'Excellent', 'Vg'],
        'Class_XII_Percentage': ['Good', 'Excellent', 'Good', 'Excellent', 'Good', 'Excellent', 'Good', 'Excellent', 'Good', 'Excellent'],
        'Class_ten_education': ['SEBA', 'SEBA', 'SEBA', 'SEBA', 'SEBA', 'CBSE', 'CBSE', 'CBSE', 'CBSE', 'CBSE'],
        'twelve_education': ['CBSE', 'CBSE', 'CBSE', 'CBSE', 'CBSE', 'AHSEC', 'AHSEC', 'AHSEC', 'AHSEC', 'AHSEC'],
        'medium': ['ENGLISH', 'ASSAMESE', 'OTHERS', 'ENGLISH', 'ASSAMESE', 'OTHERS', 'ENGLISH', 'ASSAMESE', 'OTHERS', 'ENGLISH'],
        'Father_occupation': ['BANK_OFFICIAL', 'DOCTOR', 'BANK_OFFICIAL', 'DOCTOR', 'BANK_OFFICIAL', 'DOCTOR', 'BANK_OFFICIAL', 'DOCTOR', 'BANK_OFFICIAL', 'DOCTOR'],
        'Mother_occupation': ['HOUSE_WIFE', 'DOCTOR', 'HOUSE_WIFE', 'DOCTOR', 'HOUSE_WIFE', 'DOCTOR', 'HOUSE_WIFE', 'DOCTOR', 'HOUSE_WIFE', 'DOCTOR']
    })
    y_train_fake = pd.Series(
        ['Good', 'Average', 'Vg', 'Good', 'Average', 'Vg', 'Good', 'Average', 'Vg', 'Good']
    )
    return X_train_fake, y_train_fake


def test_pipeline_builds_and_fits(sample_training_data):
    """
    Prueba que la función build_pipeline crea un pipeline
    que se puede entrenar y tiene los pasos correctos.
    """
    X_train, y_train = sample_training_data
    
    # Configuración mínima para el test
    xgb_config = {'random_state': 42} 

   # Llamar a la función a probar
    model_pipeline = train_model(xgb_config)

    # Preparar los datos
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)

    # Entrenar el pipelina
    model_pipeline.fit(X_train, y_encoded)

    # Ejecutar todas las ascersiones
    
    #  ¿Es un Pipeline? 
    assert isinstance(model_pipeline, Pipeline)

    # ¿Están los pasos correctos?
    assert 'preprocessor' in model_pipeline.named_steps
    assert 'model' in model_pipeline.named_steps
    assert isinstance(model_pipeline.named_steps['model'], XGBClassifier)

    # ¿Está el modelo entrenado?
    final_estimator = model_pipeline.named_steps['model']
    assert hasattr(final_estimator, 'classes_')
    
    # Verificar las clases (0, 1, 2)
    assert list(final_estimator.classes_) == [0, 1, 2]
    
    # Verificar también las clases originales del encoder
    assert list(label_encoder.classes_) == ['Average', 'Good', 'Vg']