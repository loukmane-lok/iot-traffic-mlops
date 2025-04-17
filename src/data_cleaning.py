from abc import ABC, abstractmethod
import pandas as pd
from typing import Union
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class DataStrategy(ABC):
    """
    Abstract base class for data handling strategies.
    """
    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series, dict]:
        """
        Abstract method to handle data processing.
        
        Args:
            data (pd.DateFrame): Input data.
        
        Returns:
            Union[pd.DataFrame, pd.Series, dict]: Processed data.
        """
        pass


class DataPreprocessingStrategy(DataStrategy):
    """
    Strategy for preprocessing data, including datetime conversion,
    sorting, and feature extraction from the datetime column.
    """
    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the data by dropping 'ID', converting 'DateTime',
        and creating time-based features.

        Args:
            data (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Preprocessed data.

        Raises:
            ValueError: If 'DateTime' column contains invalid datetime values.
            Exception: If any other error occurs during preprocessing.
        """
        try:
            # Drop ID column
            if "ID" in data.columns:
                data = data.drop(["ID"], axis=1)
            else:
                logging.warning("Column 'ID' not found in data.")

            # Convert DateTime and sort
            data['DateTime'] = pd.to_datetime(data['DateTime'], errors='coerce')
            if data['DateTime'].isnull().any():
                raise ValueError("Some values in 'DateTime' could not be converted.")

            data = data.sort_values(by=["DateTime"])

            # Feature Engineering
            data['Hour'] = data['DateTime'].dt.hour
            data['Weekday'] = data['DateTime'].dt.weekday
            data['Day'] = data['DateTime'].dt.day
            data['Month'] = data['DateTime'].dt.month
            data['Week'] = data['DateTime'].dt.isocalendar().week.astype(int)

            return data

        except Exception as e:
            logging.error(f"[DataPreprocessingStrategy] Failed to preprocess data: {e}")
            raise


class DataDivideStrategy(DataStrategy):
    """
    Strategy for dividing data into training and testing sets
    using specific features and a target column based on a date cutoff.
    """

    def handle_data(self, data: pd.DataFrame) -> dict:
        """
        Split the dataset into X_train, X_test, y_train, and y_test
        using a 4-month time-based cutoff.

        Args:
            data (pd.DataFrame): Preprocessed data containing 'DateTime',
                                 feature columns, and 'Vehicles' as the target.

        Returns:
            dict: Dictionary with keys 'X_train', 'X_test', 'y_train', 'y_test'.

        Raises:
            KeyError: If necessary columns are missing.
            Exception: If any other error occurs during splitting.
        """
        try:
            # Ensure required columns exist
            required_columns = ['DateTime', 'Vehicles', 'Hour', 'Weekday', 'Day', 'Month', 'Week']
            for col in required_columns:
                if col not in data.columns:
                    raise KeyError(f"Missing required column: {col}")

            # Define cutoff date
            cutoff_date = data['DateTime'].max() - pd.DateOffset(months=4)

            # Split data
            train_data = data[data['DateTime'] <= cutoff_date]
            test_data = data[data['DateTime'] > cutoff_date]

            # Extract features and target
            features = ['Hour', 'Weekday', 'Day', 'Month', 'Week']
            X_train = train_data[features]
            y_train = train_data['Vehicles']
            X_test = test_data[features]
            y_test = test_data['Vehicles']

            if X_train.empty or X_test.empty:
                logging.warning("[DataDivideStrategy] X_train or X_test is empty.")

            return {
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test
            }

        except Exception as e:
            logging.error(f"[DataDivideStrategy] Failed to divide data: {e}")
            raise



class DataCleaning:
    """
    Context class that uses a DataStrategy to process a dataset.
    """
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        """
        Initialize the DataCleaning context with data and a strategy.

        Args:
            data (pd.DataFrame): Input dataset.
            strategy (DataStrategy): Strategy instance to handle the data.
        """
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series, dict]:
        """
        Apply the provided strategy to the dataset.

        Returns:
            Union[pd.DataFrame, pd.Series, dict]: Result after applying the strategy.

        Raises:
            Exception: If an error occurs during strategy execution.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error(f"[DataCleaning] Error in handling the data: {e}")
            raise
