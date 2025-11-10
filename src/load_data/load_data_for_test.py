import logging
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga un archivo CSV desde una ruta y devuelve un DataFrame.
    Lanza FileNotFoundError si no se encuentra.
    """
    logger.info(f"Cargando DataFrame desde {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info("DataFrame cargado exitosamente.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Archivo no encontrado en {file_path}")
        raise