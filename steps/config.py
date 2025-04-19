from zenml.steps import BaseParameters
from typing import List

class ModelNameConfig(BaseParameters):
    """
    Model config
    """
    model_name:str = "XGBRegressor"
    features: List[str]
    target: str = "Vehicles"