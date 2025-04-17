import os
import logging
import pandas as pd
from zenml import step

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class DataIngest:
    """
    Class responsible for ingesting data from a specified CSV file path.
    
    Attributes:
        data_path (str): The path to the CSV file containing the dataset.
    """
    def __init__(self, data_path: str):
        """
        Initialize the DataIngest object with a given file path.

        Args:
            data_path (str): Path to the CSV file.
        """
        self.data_path = data_path
        
    def get_data(self) -> pd.DataFrame:
        """
        Load and return data from the CSV file.

        Returns:
            pd.DataFrame: The loaded dataset.

        Raises:
            FileNotFoundError: If the CSV file does not exist.
            pd.errors.ParserError: If the file cannot be parsed as a CSV.
        """
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"File not found: {self.data_path}")
        logging.info(f"Ingesting data from: {self.data_path}")
        return pd.read_csv(self.data_path)
    
@step 
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    ZenML step to ingest data from a given path using the DataIngest class.

    Args:
        data_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: The ingested dataset.

    Raises:
        RuntimeError: If there is an issue during the ingestion process.
    """
    try:
        logging.info(f"Ingestion Step")
        data_ingest = DataIngest(data_path)
        data = data_ingest.get_data()
        return data
    except Exception as e:
        logging.error(f"Error while ingesting the data: {e}")
        raise RuntimeError(f"Data ingestion failed for {data_path}") from e
    