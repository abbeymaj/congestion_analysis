# Importing packages
import os
import pytest
import pandas as pd
from src.components.config_entity import DataIngestionConfig

# Creating a function to get the path to the train dataset
@pytest.fixture(scope='function')
def train_dataset_path():
    train_data_config = DataIngestionConfig()
    return train_data_config.train_data_path

# Creating a function to get the path to the test dataset
@pytest.fixture(scope='function')
def dataset_test_path():
    test_data_config = DataIngestionConfig()
    return test_data_config.test_data_path

# Verify if the artifacts directory exists
def test_artifacts_directory_exists():
    assert os.path.exists('artifacts') is True

# Verify that the train dataset is present in the artifacts folder
def test_train_dataset_exists(train_dataset_path):
    assert os.path.exists(train_dataset_path) is True

# Verify that the train dataset is not empty
def test_train_dataset_not_empty(train_dataset_path):
    assert os.path.getsize(train_dataset_path) > 0

# Verify that the test dataset is present in the artifacts folder
def test_test_dataset_exists(dataset_test_path):
    assert os.path.exists(dataset_test_path) is True

# Verify that the test dataset is not empty
def test_test_dataset_not_empty(dataset_test_path):
    assert os.path.getsize(dataset_test_path) > 0

# Verify that the train dataset has 5 columns
def test_train_dataset_column_count(train_dataset_path):
    df = pd.read_parquet(train_dataset_path)
    assert len(list(df.columns)) == 5

# Verify that the test dataset has 5 columns
def test_test_dataset_column_count(dataset_test_path):
    df = pd.read_parquet(dataset_test_path)
    assert len(list(df.columns)) == 5