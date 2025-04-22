from pipeline.training_pipeline import training_pipeline

from steps.config import ModelNameConfig

model = "XGBRegressor"
features = ['Hour', 'Weekday', 'Day', 'Month', 'Week']
target = "Vehicles"

config = ModelNameConfig(model=model, features=features, target=target)

data_path = 'data/train_ML_IOT.csv'

if __name__ == "__main__":
    training_pipeline(data_path=data_path, )