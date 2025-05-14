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

# Reading the test dataset path
@pytest.fixture(scope='function')
def dataset_test_path():
    test_data_config = DataIngestionConfig()
    return test_data_config.test_data_path

# Verifying that the generate_features method works as expected
def test_generate_features_function(train_dataset_path):
    df = pd.read_parquet(train_dataset_path)
    transform = DataTransformation()
    transformed_df = transform.generate_features(df)
    assert transformed_df is not None
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

# Verifying that the preprocessor object is created
def test_create_preprocessor_obj():
    transform = DataTransformation()
    preprocessor = transform.create_preprocessor_obj()
    assert preprocessor is not None

# Verifying that the initiate_data_transformation method works
# as expected
def test_initiate_data_transformation(train_dataset_path, dataset_test_path):
    transform = DataTransformation()
    train_set, test_set = transform.initiate_data_transformation(train_dataset_path, dataset_test_path, save_object=False)
    assert train_set is not None
    assert test_set is not None