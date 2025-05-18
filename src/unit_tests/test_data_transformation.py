# Importing packages
import os
import pytest
import pandas as pd
from src.components.config_entity import DataIngestionConfig
from src.components.config_entity import DataTransformationConfig
from src.components.config_entity import StoreFeatureConfig
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

# Reading the transformed train dataset path
@pytest.fixture(scope='function')
def xform_train_dataset_path():
    store_config = StoreFeatureConfig()
    return store_config.xform_train_path

# Reading the transformed test dataset path
@pytest.fixture(scope='function')
def xform_test_dataset_path():
    store_config = StoreFeatureConfig()
    return store_config.xform_test_path

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

# Verifying that the preprocessor object is present in the artifacts directory
def test_preprocessor_object_exists():
    preprocessor_config = DataTransformationConfig()
    preprocessor_obj_path = preprocessor_config.preprocessor_obj_path
    assert os.path.exists(preprocessor_obj_path) is True

# Verifying that the feature store directory is created
def test_feature_store_directory_exists():
    assert os.path.exists('feature_store') is True

# Verifying that the transformed train dataset is present in the feature store
def test_xform_train_dataset_exists(xform_train_dataset_path):
    assert os.path.exists(xform_train_dataset_path) is True

# Verifying that the transformed train dataset is not empty
def test_xform_train_dataset_not_empty(xform_train_dataset_path):
    assert os.path.getsize(xform_train_dataset_path) > 0

# Verifying that the transformed test dataset is present in the feature store
def test_xform_test_dataset_exists(xform_test_dataset_path):
    assert os.path.exists(xform_test_dataset_path) is True

# Verifying that the transformed test dataset is not empty
def test_xform_test_dataset_not_empty(xform_test_dataset_path):
    assert os.path.getsize(xform_test_dataset_path) > 0

# Verifying that the target column is present in the transformed train dataset
def test_target_column_present_in_trainset(xform_train_dataset_path):
    df = pd.read_parquet(xform_train_dataset_path)
    col_list = list(df.columns)
    assert 'congestion' in col_list

# Verifying that the target column is present in the transformed test dataset
def test_target_column_present_in_testset(xform_test_dataset_path):
    df = pd.read_parquet(xform_test_dataset_path)
    col_list = list(df.columns)
    assert 'congestion' in col_list 
    