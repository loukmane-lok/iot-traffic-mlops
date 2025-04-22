import logging
import pandas as pd
from zenml import step
from src.data_cleaning import DataCleaning, DataPreprocessingStrategy, DataDivideStrategy
from typing import Tuple

@step
def clean_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean and prepare the dataset by applying preprocessing and train-test split strategies.

    Args:
        data (pd.DataFrame): Raw input dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: train_data and test_data.
    """
    try:
        logging.info("Data Cleaning Step ...")
        # Apply preprocessing strategy
        preprocessing = DataCleaning(data, DataPreprocessingStrategy())
        processed_data = preprocessing.handle_data()

        # Apply data division strategy
        divider = DataCleaning(processed_data, DataDivideStrategy())
        data_dict = divider.handle_data()
        logging.info("Data Cleaning Completed !")
        
        # Return train and test data separately
        return data_dict['train_data'], data_dict['test_data']

    except Exception as e:
        logging.error(f"[clean_data] Failed to clean and divide the data: {e}")
        raise

        