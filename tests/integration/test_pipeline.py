import pytest
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.load_data.load_data_for_test import load_data
from src.clean_data.clean_data_for_test import clean_data
from src.train_test_split_data.train_test_split_data_for_test import train_test_split_data
from src.train_model.train_model_for_test import train_model
from src.test_model.evaluate_model_for_test import evaluate_model


@pytest.fixture(scope="module")
def sample_dirty_data_file(tmp_path_factory):
    """
    Crea un archivo CSV "sucio" de prueba en un directorio temporal.
    """
    # Definir datos sucios (20 filas para un split 75/25 -> 15/5)
    data = {
        'Performance': ['gOOD'] * 5 + ['vG'] * 5 + ['eXCELLENT'] * 10,
        'Gender': ['MALE'] * 10 + ['FEMALE'] * 10,
        'Caste': ['st', 'obc', 'gENERAL', 'sc', 'st'] * 4,
        'coaching': ['no', 'no', 'wa', 'wa'] * 5,
        'time': ['one', 'one', 'one', 'one', 'one', 'two', 'two', 'two', 'two', 'two'] * 2,
        'Class_ X_Percentage': ['eXCELLENT', 'vG', 'eXCELLENT', 'vG', 'eXCELLENT', 'vG', 'eXCELLENT', 'vG', 'eXCELLENT', 'vG'] * 2,
        'Class_XII_Percentage': ['gOOD', 'eXCELLENT', 'gOOD', 'eXCELLENT', 'gOOD', 'eXCELLENT', 'gOOD', 'eXCELLENT', 'gOOD', 'eXCELLENT'] * 2,
        'Class_ten_education': ['seba', 'seba', 'seba', 'seba', 'seba', 'cbse', 'cbse', 'cbse', 'cbse', 'cbse'] * 2,
        'twelve_education': ['cbse', 'cbse', 'ahsec', 'ahsec', 'ahsec'] * 4,
        'medium': ['english', 'assamese', 'others', 'english'] * 5,
        'Father_occupation': ['business', 'doctor', 'others', 'bank_official', 'engineer'] * 4,
        'Mother_occupation': ['house_wife', 'doctor'] * 10
    }
    df = pd.DataFrame(data)
    
    # Crear el archivo CSV
    tmp_dir = tmp_path_factory.mktemp("integration_data")
    file_path = tmp_dir / "dirty_data.csv"
    df.to_csv(file_path, index=False)
    
    # Retornar la ruta y el nombre de la columna objetivo
    return str(file_path), 'Performance'


def test_end_to_end(sample_dirty_data_file):
    """
    Prueba el flujo completo del pipeline:
    load_data -> clean_data -> train_test_split -> train_model -> test_model
    """
    
    file_path, target_column = sample_dirty_data_file
    
   # load_data
    raw_data = load_data(file_path)
    assert not raw_data.empty, "load_data devolvió un DataFrame vacío"

    # clean_data
    cleaned_data = clean_data(raw_data)
    assert not cleaned_data.empty, "clean_data devolvió un DataFrame vacío"
        
    # train_test_split
    train_df, test_df = train_test_split_data(
        cleaned_data, 
        test_size=0.25, 
        random_state=42, 
        stratify_column=target_column
    )
    assert not train_df.empty, "train_df está vacío después del split"
    assert not test_df.empty, "test_df está vacío después del split"

    # Separar X/y
    assert target_column in train_df.columns
    assert target_column in test_df.columns
    
    X_train = train_df.drop(columns=[target_column])
    y_train_raw = train_df[target_column]
    X_test = test_df.drop(columns=[target_column])
    y_test_raw = test_df[target_column]

    # Codificar 'y'
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train_raw)
    y_test = label_encoder.transform(y_test_raw)
        
    # train_model
    xgb_config = {'random_state': 42} 
    pipeline = train_model(xgb_config)
        
    # Entrenar
    pipeline.fit(X_train, y_train)

    # Predecir
    y_pred = pipeline.predict(X_test)
        
    # Evaluar
    metrics = evaluate_model(y_test, y_pred)
    
    # Asserts finales
    assert metrics is not None
    assert isinstance(metrics, dict)
    assert 'accuracy' in metrics
    assert 'f1_macro' in metrics
    assert metrics['accuracy'] >= 0.0
    assert len(X_train) == 15 # 75% de 20
    assert len(X_test) == 5  # 25% de 20