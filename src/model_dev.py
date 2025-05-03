from abc import ABC, abstractmethod
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
import numpy as np
import pandas as pd
import logging

def rmse_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

rmse_scorer = make_scorer(rmse_scorer, greater_is_better=False)


class Model(ABC):
    """
    Abstract model class.
    """
    @abstractmethod
    def train(self, train_data):
        """
        Abstract method to train a model.
        
        Args:
            train_data (pd.DataFrame): Training data.
        
        Returns:
            Trained model.
        """
        pass
    
class XGBRegressorModel(Model):
    """
    XGBRegressor model with hyperparameter tuning.
    """
    
    def __init__(self, param_grid=None, n_splits=3, random_state=42):
        self.param_grid = param_grid or {
            'n_estimators': [25, 50, 75],                 # fewer trees
            'max_depth': [2, 3],                      # shallower trees
            'learning_rate': [0.01, 0.05],            # slower learning
            'subsample': [0.6, 0.8],                  # use part of data for each tree
            'colsample_bytree': [0.6, 0.8],           # use part of features
            'reg_alpha': [0.1, 1],                    # L1 regularization
            'reg_lambda': [1, 5]                      # L2 regularization
        }
        self.n_splits = n_splits
        self.random_state = random_state
        self.best_models = {}
        self.best_params = {}
        self.best_scores = {}
        
    def train(self, train_data: pd.DataFrame, features: list[str], target: str = 'Vehicles') -> dict:
        try:
            logging.info("Training a single model for all junctions...")

            X = train_data[features]
            y = train_data[target]

            tcvs = TimeSeriesSplit(n_splits=self.n_splits)

            model = XGBRegressor(random_state=self.random_state)
            grid = GridSearchCV(model, self.param_grid, scoring=rmse_scorer, cv=tcvs, verbose=0)
            grid.fit(X, y)

            best_model = grid.best_estimator_
            best_params = grid.best_params_
            best_score = grid.best_score_

            logging.info(f"Global Model RMSE = {-best_score:.2f}")
            logging.debug(f"Best Params: {best_params}")

            return {
                "model": best_model,
                "params": best_params,
                "score": best_score
            }

        except Exception as e:
            logging.error(f"Failed to train global model: {e}")
            raise e
