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
    Trains a specified model using the provided configuration and training data.
    Logs training parameters and models using Mlflow nested runs.

    Args:
        train_data (pd.DataFrame): The training dataset.
        config (ModelNameConfig): Configuration containing model name, features, and target.

    Returns:
        dict: Trained models for each junction.
    """
    try:
        logging.info("Training Step ...")

        if config.model != "XGBRegressor":
            raise ValueError(f"Model {config.model} not supported")

        model = XGBRegressorModel()
        trained_models = {}

        junctions = sorted(train_data['Junction'].unique())
        for junc in tqdm(junctions, desc="Training models by junction"):
            start_time = time.time()
            subset = train_data[train_data['Junction'] == junc].copy()

            # Fix integer + NaN issue
            for col in config.features:
                if subset[col].dtype == "int" and subset[col].isnull().any():
                    subset[col] = subset[col].astype("float64")

            with mlflow.start_run(run_name=f"train_junction_{junc}", nested=True):
                results = model.train(subset, features=config.features, target=config.target)
                mlflow.log_params(results.get("params", {}).get(str(junc), {}))
                model_to_log = results["models"][str(junc)]
                mlflow.xgboost.log_model(model_to_log, f"model_junction_{junc}")
                trained_models[str(junc)] = model_to_log

            elapsed = time.time() - start_time
            logging.info(f"Junction {junc} trained in {elapsed:.2f} seconds.")

        logging.info("Training Step Completed!")
        return trained_models

    except Exception as e:
        logging.error(f"Error in training the model: {e}")
        raise
