from zenml import step
import pandas as pd
from src.model_dev import XGBRegressorModel
from steps.config import ModelNameConfig
import logging

@step(enable_cache=False)
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
        logging.info("Training Step ...")
        model = None
        if config.model == "XGBRegressor":
            model = XGBRegressorModel()
            results = model.train(train_data, features=config.features, target=config.target)
            
            trained_models = {str(junc): model for junc, model in results["models"].items()}
            
            logging.info("Training Step Completed !")
            
            return trained_models
        else:
            raise ValueError(f"Model {config.model} not supported")
    except Exception as e:
        logging.error(f"Error in training the model: {e}")
        raise