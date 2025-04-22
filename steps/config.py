from zenml.config.base_settings import BaseSettings
from typing import List

class ModelNameConfig(BaseSettings):
    """ Model config """
    model: str = "XGBRegressor"
    features: List[str]
    target: str = "Vehicles"