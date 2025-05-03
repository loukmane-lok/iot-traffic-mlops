from zenml import step
from src.evaluation import RMSE
import pandas as pd
from typing import Dict
from steps.config import ModelNameConfig
import logging
import mlflow
from steps.utils import experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def evaluate_model(test_data: pd.DataFrame, 
                   trained_model: object, 
                   config: ModelNameConfig
                ) -> Dict[str, float]:
    """
    Evaluate the trained global model on the entire test dataset.

    Args:
        test_data (pd.DataFrame): Test dataset.
        trained_model (object): Trained global model.
        config (ModelNameConfig): Model configuration including features and target.
        
    Returns: 
        Dict[str, float]: RMSE score for the global model.
    """
    try:
        logging.info("Evaluation Step ...")
        rmse = RMSE()

        X_test = test_data[config.features]
        y_test = test_data[config.target]

        y_pred = trained_model.predict(X_test)
        rmse_score = rmse.calculate_scores(y_test, y_pred)

        mlflow.log_metric("rmse", rmse_score)
        logging.info(f"Global Model RMSE = {rmse_score:.2f}")

        logging.info("Evaluation Step Completed!")
        return {"rmse": rmse_score}

    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        raise
