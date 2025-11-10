import pytest
import pandas as pd
import numpy as np

from src.train_test_split_data.train_test_split_data_for_test import train_test_split_data

@pytest.fixture
def sample_clean_data():
    """
    Crea un DataFrame de 100 filas para probar la división.
    Incluye un objetivo binario desbalanceado (80/20)
    para poder probar la estratificación.
    """
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        # Creamos un objetivo (target) con 80% clase 0 y 20% clase 1
        'target': [0] * 80 + [1] * 20
    }
    df = pd.DataFrame(data)
    
    # Mezclar DaraFrame para que las clases no estén ordenadas
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df


def test_split_data_ratios(sample_clean_data):
    """
    Prueba que la división 80/20 se aplica correctamente.
    (100 filas -> 80 train, 20 test)
    """
    df = sample_clean_data
    test_size = 0.2
    
    train_df, test_df = train_test_split_data(
        df,
        test_size=test_size,
        random_state=42,
        stratify_column='target'
    )
    
    assert len(train_df) == 80
    assert len(test_df) == 20
    

def test_no_overlap(sample_clean_data):
    """
    Prueba que no haya fuga de datos.
    Verifica que los índices de train y test sean conjuntos disjuntos.
    """
    df = sample_clean_data
    
    train_df, test_df = train_test_split_data(
        df, 
        test_size=0.2, 
        random_state=42, 
        stratify_column='target'
    )
    
    # Los DataFrames de pandas retienen su índice original después del split
    train_indices = set(train_df.index)
    test_indices = set(test_df.index)
    
    assert train_indices.isdisjoint(test_indices)
    
    
def test_stratification(sample_clean_data):
    """
    Prueba que la estratificación funciona.
    La proporción de clases (80/20) debe mantenerse
    en los conjuntos de train y test.
    """
    df = sample_clean_data
    # Proporción original de la clase 1 es 20 / 100 = 0.2
    original_proportion = df['target'].value_counts(normalize=True)[1]
    
    train_df, test_df = train_test_split_data(
        df,
        test_size=0.2,
        random_state=42,
        stratify_column='target'
    )
    
    # Verificamos la proporción de la clase 1 en 'y_train'
    # 80 filas de train * 0.2 = 16 filas deben ser clase 1
    train_proportion = train_df['target'].value_counts(normalize=True)[1]
    
    # Verificamos la proporción de la clase 1 en 'y_test'
    # 20 filas de test * 0.2 = 4 filas deben ser clase 1
    test_proportion = test_df['target'].value_counts(normalize=True)[1]
    
    # Usamos pytest.approx para manejar la comparación de floats
    assert train_proportion == pytest.approx(original_proportion)
    assert test_proportion == pytest.approx(original_proportion)