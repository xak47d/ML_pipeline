import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica la limpieza de datos a un DataFrame.
    """
    logger.info("Iniciando lógica de limpieza de datos...")
    
    # Trabajar sobre una copia para no modificar el original
    df_cleaned = df.copy()

    # Definir un mapa de limpieza centralizado
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
        if col in df_cleaned.columns:
            # Encadenar .str.strip() y .replace()
            df_cleaned[col] = df_cleaned[col].str.strip().replace(replacements)

    # Reemplazar valores nulos genéricos
    df_cleaned.replace(['', ' ', 'NAN'], np.nan, inplace=True)

    # Obtener la lista de columnas que acabamos de limpiar
    cols_to_impute = [col for col in cleaning_map.keys() if col in df_cleaned.columns]

    # Calcular todas las modas necesarias de una vez
    if cols_to_impute:
        imputation_values = {}
        for col in cols_to_impute:
            mode_val = df_cleaned[col].mode()
            if not mode_val.empty:
                imputation_values[col] = mode_val.loc[0]
        
        # Aplicar fillna UNA SOLA VEZ con el diccionario de modas
        df_cleaned.fillna(imputation_values, inplace=True)

    # Eliminar columna no deseada
    col_to_drop = 'mixed_type_col'
    if col_to_drop in df_cleaned.columns:
        df_cleaned.drop(col_to_drop, axis=1, inplace=True)

    logger.info("Lógica de limpieza finalizada.")
    return df_cleaned