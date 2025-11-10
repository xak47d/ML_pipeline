import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger(__name__)

def train_test_split_data(df: pd.DataFrame, 
                    test_size: float, 
                    random_state: int, 
                    stratify_column: str = None):
    """
    Toma un DataFrame y lo divide en train y test.
    """
    logger.info("Dividiendo DataFrame en train y test...")
    
    stratify_data = None
    if stratify_column and stratify_column in df.columns:
        stratify_data = df[stratify_column]
        
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_data
    )
    
    return train_df, test_df