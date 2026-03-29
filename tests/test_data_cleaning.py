import pandas as pd
import pytest
from src.data_cleaning import DataPreprocessingStrategy

def test_data_preprocessing_strategy():
    # Setup: Create a sample dataframe
    df = pd.DataFrame({
        'DateTime': ['2023-11-20 10:00:00'],
        'ID': [1],
        'Vehicles': [10]
    })
    
    strategy = DataPreprocessingStrategy()
    processed_df = strategy.handle_data(df)
    
    # Assertions
    assert 'ID' not in processed_df.columns
    assert 'Hour' in processed_df.columns
    assert 'Weekday' in processed_df.columns
    assert processed_df.iloc[0]['Hour'] == 10
    assert processed_df.iloc[0]['Weekday'] == 0  # 2023-11-20 was Monday (0)
