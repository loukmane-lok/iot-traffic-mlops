from zenml import step
import pandas as pd
from model_dev import XGBRegressorModel
from config import ModelNameConfig
import logging

@step
def train_model(train_data: pd.DataFrame, config: ModelNameConfig):
    """
    Trains a specified model using the provided configuration and training data.

    Args:
        train_data (pd.DataFrame): The training dataset.
        config (ModelNameConfig): Configuration containing model name, features, and target.

    Returns:
        dict: Trained models for each junction.
    """
    try:
        model = None
        if config.model_name == "XGBRegressor":
            model = XGBRegressorModel()
            results = model.train(train_data, features=config.features, target=config.target)
            trained_models = results["models"]
            return trained_models
        else:
            raise ValueError(f"Model {config.model_name} not supported")
    except Exception as e:
        logging.error(f"Error in training the model: {e}")
        raise