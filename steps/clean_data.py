import logging
import pandas as pd
from zenml import step
from data_cleaning import DataCleaning, DataPreprocessingStrategy, DataDivideStrategy


@step
def clean_data(data: pd.DataFrame) -> dict:
    """
    Clean and prepare the dataset by applying preprocessing and train-test split strategies.

    Args:
        data (pd.DataFrame): Raw input dataset.

    Returns:
        dict: Dictionary containing train_data, test_data.
    """
    try:
        # Apply preprocessing strategy
        preprocessing = DataCleaning(data, DataPreprocessingStrategy())
        processed_data = preprocessing.handle_data()

        # Apply data division strategy
        divider = DataCleaning(processed_data, DataDivideStrategy())
        data_dict = divider.handle_data()
        logging.info("Data Cleaning Completed !")
        return data_dict

    except Exception as e:
        logging.error(f"[clean_data] Failed to clean and divide the data: {e}")
        raise 

        