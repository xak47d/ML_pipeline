import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from src.clean_data.clean_data_for_test import clean_data

@pytest.fixture
def data_fixture():
    """
    Fixture que crea un DataFrame "sucio" de entrada y
    el DataFrame "limpio" que esperamos como salida.
    """
    
    # Mezcla de valores para limpiar
    dirty_df = pd.DataFrame({
        'Performance': ['gOOD', 'aVERAGE', 'vG', 'Good'],
        'Gender': ['MALE', 'FEMALE', 'MALE', 'FEMALE'],
        'Caste': ['st', 'obc', 'gENERAL', 'st'],
        'Father_occupation': ['business', 'doctor', 'others', 'bank_official'],
        'mixed_type_col': [1, 'a', 3.0, None]
    })
    
    # Resultado Esperado
    expected_clean_df = pd.DataFrame({
        'Performance': ['Good', 'Average', 'Vg', 'Good'], # 'Good' se mantiene
        'Gender': ['male', 'female', 'male', 'female'],
        'Caste': ['ST', 'OBC', 'General', 'ST'],
        'Father_occupation': ['BUSINESS', 'DOCTOR', 'OTHERS', 'BANK_OFFICIAL']
    })
    
    return dirty_df, expected_clean_df
    
    
def test_mapping_correctos(data_fixture):
    """
    Prueba que los mapeos de estandarización se aplican
    correctamente a las columnas.
    """
    
    # Preparar datos
    dirty_df, expected_df = data_fixture
    
    # Ejecutar la función
    cleaned_df = clean_data(dirty_df)
    
    # Comparar si el resultado es idéntico a lo que esperamos
    assert_frame_equal(cleaned_df, expected_df)
   
    
def test_no_modifica_original(data_fixture):
    """
    Prueba que la función no modifica
    el DataFrame original.
    """
    
    dirty_df, _ = data_fixture
    
    # Se crea una copia para hacer la comparación
    original_df_before = dirty_df.copy()
    
    # Llamar la función
    clean_data(dirty_df)
    
    # Verificar que el DataFrame original 'dirty_df' sigue siendo igual que antes
    assert_frame_equal(dirty_df, original_df_before)