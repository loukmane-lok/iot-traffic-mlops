from zenml import step
import pandas as pd

@step
def evaluate_model(data: pd.DataFrame):
    pass