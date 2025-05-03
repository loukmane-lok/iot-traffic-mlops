from pipeline.training_pipeline import training_pipeline

from steps.config import ModelNameConfig
from zenml.client import Client
model = "XGBRegressor"
features = ['Hour', 'Weekday', 'Day', 'Month', 'Week']
target = "Vehicles"

config = ModelNameConfig(model=model, features=features, target=target)

data_path = 'data/train_ML_IOT.csv'

if __name__ == "__main__":
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    training_pipeline(data_path=data_path, config=config)
    
# mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri file:///path/to/mlruns
# mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri http://username:password@host:port/mlruns
    
    