from zenml import step
from src.evaluation import RMSE
import pandas as pd
from typing import Dict
from steps.config import ModelNameConfig
import logging

@step(enable_cache=False)
def evaluate_model(test_data: pd.DataFrame, 
                   trained_models: Dict[str, object], 
                   config: ModelNameConfig
                ) -> Dict[str, float]:
    """
    Evaluate trained models.
    
    Args:
        test_data (pd.DataFrame): Test dataset.
        trained_models(Dict[str, object]): Dictionary of trained models per junction.
        config(ModelNameConfig): Model configuration including features and target.
        
    Returns: 
        Dict[str, float]: RMSE score per junction.
    """
    try:
        logging.info("Evaluation Step ...")
        rmse = RMSE()
        scores = {}
        
        for junc in sorted(test_data['Junction'].unique()):
            
            subset = test_data[test_data['Junction'] == junc].copy()
            model = trained_models.get(str(junc))
            
            if model is None:
                logging.warning(f"No trained model found for junction {junc}")
                continue
            
            X_test = subset[config.features]
            y_test = subset[config.target]
            
            y_pred = model.predict(X_test)

            rmse_score = rmse.calculate_scores(y_test, y_pred)
            
            scores[junc] = rmse_score
            
            logging.info(f"Junction {junc}: RMSE = {rmse_score:.2f}")

        logging.info("Evaluation Step Completed !")
        return scores
        
    except Exception as e:
        logging.error(f"Error during evaluation {e}")
        raise