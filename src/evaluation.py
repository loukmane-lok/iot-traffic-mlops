import logging
from abc import ABC, abstractmethod
from sklearn.metrics import mean_squared_error
import numpy as np

class Evaluation(ABC):
    """
    Abstract base class for evaluation metrics.
    """
    @abstractmethod
    def calculate_scores(self, y_true, y_pred):
        """
        Calculate evaluation metric.

        Args:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted target values.

        Returns:
            float: Evaluation score.
        """
        pass 

class RMSE(Evaluation):
    """
    Root Mean Squared Error evaluator.
    """
    def calculate_scores(self, y_true, y_pred):
        try:
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            return rmse
        except Exception as e:
            logging.error(f"Error calculating RMSE: {e}")
            raise
