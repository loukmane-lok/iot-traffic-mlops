from zenml import step
import pandas as pd
from src.model_dev import XGBRegressorModel
from steps.config import ModelNameConfig
from steps.utils import experiment_tracker
import logging
import mlflow
from tqdm import tqdm
import time

@step(experiment_tracker=experiment_tracker.name)
def train_model(train_data: pd.DataFrame, config: ModelNameConfig):
    """
    Trains a single model across all junctions using the provided configuration and training data.
    Logs training parameters and the model using MLflow.

    Args:
        train_data (pd.DataFrame): The training dataset.
        config (ModelNameConfig): Configuration containing model name, features, and target.

    Returns:
        Trained model object.
    """
    try:
        logging.info("Training single global model...")

        if config.model != "XGBRegressor":
            raise ValueError(f"Model {config.model} not supported")

        model = XGBRegressorModel()

        # Fix integer + NaN issue globally
        for col in config.features:
            if train_data[col].dtype == "int" and train_data[col].isnull().any():
                train_data[col] = train_data[col].astype("float64")

        start_time = time.time()

        with tqdm(total=1, desc="Training Global Model") as pbar:
            with mlflow.start_run(run_name="train_global_model", nested=True):
                results = model.train(train_data, features=config.features, target=config.target)
                mlflow.log_params(results.get("params", {}))
                mlflow.xgboost.log_model(results["model"], "global_model")
                trained_model = results["model"]
            pbar.update(1)

        elapsed = time.time() - start_time
        logging.info(f"Global model trained in {elapsed:.2f} seconds.")
        logging.info("Training Step Completed!")

        return trained_model

    except Exception as e:
        logging.error(f"Error in training the model: {e}")
        raise
