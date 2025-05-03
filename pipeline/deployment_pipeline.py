import numpy as np
import pandas as pd
from steps.clean_data import clean_data
from steps.evaluate_model import evaluate_model
from steps.ingest_data import ingest_data
from steps.train_model import train_model
from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
from zenml.integrations.constants import MLFLOW, TENSORFLOW
from zenml.integrations.mlflow.model_deployers.mlflow_model_deployer import MLFlowModelDeployer
from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.config.base_settings import BaseSettings
from steps.config import ModelNameConfig


docker_settings = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseSettings):
    """Holds parameters for deployment trigger."""
    max_rmse: float = 50.0

class MLFlowDeploymentLoaderStepParameter(BaseSettings):
    """
    MLFlow Deployment parameters.
    """
    pipeline_name: str
    step_name: str
    running: bool = True

@step
def deployment_trigger(rmse: float, config: DeploymentTriggerConfig) -> bool:
    """Triggers deployment if RMSE is below the threshold."""
    return rmse < config.max_rmse

@step
def check_rmse_and_trigger(scores: dict, config: DeploymentTriggerConfig) -> bool:
    rmse = scores["rmse"]
    return rmse < config.max_rmse

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def continuous_deployment_pipeline(max_rmse: float = 50.0, workers: int = 4):
    # Define model config
    model = "XGBRegressor"
    features = ['Hour', 'Weekday', 'Day', 'Month', 'Week']
    target = "Vehicles"
    config = ModelNameConfig(model=model, features=features, target=target)

    # Step 1: Load and clean data
    raw_data = ingest_data(data_path="/home/lok/Documents/ML_Projects/iot-traffic-mlops/data/train_ML_IOT.csv")  
    train_data, test_data = clean_data(data=raw_data)

    # Step 2: Train and evaluate
    trained_model = train_model(train_data=train_data, config=config)
    scores = evaluate_model(test_data=test_data, trained_model=trained_model, config=config)

    # This step reads the score and decides deployment
    deploy_decision = check_rmse_and_trigger(scores=scores, config=DeploymentTriggerConfig(max_rmse=max_rmse))

    mlflow_model_deployer_step(
        model=trained_model,
        deploy_decision=deploy_decision,
        workers=workers
    )
    

