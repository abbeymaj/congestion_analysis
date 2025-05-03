# Importing packages
import os
import pytest
import pandas as pd
from src.components.config_entity import DataIngestionConfig
from src.components.data_transformation import DataTransformation

# Reading the train dataset path
@pytest.fixture(scope='function')
def train_dataset_path():
    train_data_config = DataIngestionConfig()
    return train_data_config.train_data_path

# Verifying that the generate_features method works as expected
def test_generate_features_function(train_dataset_path):
    df = pd.read_parquet(train_dataset_path)
    transform = DataTransformation()
    transformed_df = transform.generate_features(df)
    assert 'hour_sin' in list(transformed_df.columns)
    assert 'hour_cos' in list(transformed_df.columns)
    assert 'am_pm' in list(transformed_df.columns)
    assert 'is_weekend' in list(transformed_df.columns)
    assert 'x_y_direction' in list(transformed_df.columns)
    assert 'hour' not in list(transformed_df.columns)
    assert 'x' not in list(transformed_df.columns)
    assert 'y' not in list(transformed_df.columns)
    assert 'direction' not in list(transformed_df.columns)
    assert 'time' not in list(transformed_df.columns)