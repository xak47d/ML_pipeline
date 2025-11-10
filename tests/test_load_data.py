import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from src.load_data.load_data_for_test import load_data

@pytest.fixture
def archivo_csv_falso(tmp_path):
    """
    Crea un archivo CSV falso en un directorio temporal.
    """
    
    # Definir los datos de prueba
    datos_falsos = {
        'input1': [1, 2, 3, 4],
        'input2': [1.5, 2.5, 3.5, 4.5],
        'output': [0, 1, 0, 1]
    }
    # Definir los datos de prueba
    expected_df = pd.DataFrame(datos_falsos)
    
    # Crear la ruta del archivo dentro del directorio temporal
    file_path = tmp_path / "datos.csv"
    
    # Guardar el DataFrame falso como CSV en esa ruta
    expected_df.to_csv(file_path, index=False)
    
    # Entregar la ruta y el DataFrame esperado a la prueba
    return str(file_path), expected_df


def test_load_data_exito(archivo_csv_falso):
    """
    Prueba que la función cargue correctamente un archivo CSV válido.
    """
    
    # Obtenemos la ruta y el DataFrame esperado de nuestra fixture
    file_path, expected_df = archivo_csv_falso
    
    # Llamamos a la función que estamos probando
    loaded_df = load_data(file_path)
    
    # Verificamos que el resultado sea el esperado
    assert isinstance(loaded_df, pd.DataFrame)
    assert_frame_equal(loaded_df, expected_df)
 
    
def test_load_data_not_found():
    """
    Prueba que se lance FileNotFoundError si el archivo no existe.
    """
    non_existent_path = "ruta/que/no/existe.csv"
    
    # Falla de carga del archivo
    with pytest.raises(FileNotFoundError):
        load_data(non_existent_path)