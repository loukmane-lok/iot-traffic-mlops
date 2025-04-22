from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.clean_data import clean_data
from steps.train_model import train_model
from steps.evaluate_model import evaluate_model
from steps.config import ModelNameConfig

model = "XGBRegressor"
features = ['Hour', 'Weekday', 'Day', 'Month', 'Week']
target = "Vehicles"

config = ModelNameConfig(model=model, features=features, target=target)

@pipeline
def training_pipeline(data_path: str, config: ModelNameConfig=config):
    """
    Full training pipeline that ingests, cleans, trains, and evaluates a model.

    Args:
        data_path (str): Path to the dataset CSV file.
        config (ModelNameConfig): Configuration for training and evaluation.
    """
    # Step 1: Load raw data
    raw_data = ingest_data(data_path=data_path)

    # Step 2: Clean and split data
    train_data, test_data = clean_data(data=raw_data)


    # Step 3: Train model
    trained_models = train_model(train_data=train_data, config=config)

    # Step 4: Evaluate model
    scores = evaluate_model(test_data=test_data, trained_models=trained_models, config=config)

    return scores
   
    
    