# Importing packages
import os
import pytest
import pandas as pd
from xgboost import XGBRegressor
from src.components.config_entity import StoreFeatureConfig
from src.components.model_trainer import ModelTrainer
from src.components.find_best_model import FindBestModel

# Reading the transformed train dataset path
@pytest.fixture(scope='function')
def xform_train_dataset_path():
    train_config = StoreFeatureConfig()
    return train_config.xform_train_path

# Reading the transformed test dataset path
@pytest.fixture(scope='function')
def xform_test_dataset_path():
    test_config = StoreFeatureConfig()
    return test_config.xform_test_path

# Verifying that the feature and target datasets can be created
def test_create_feature_target_datasets():
    trainer = ModelTrainer()
    X_train, y_train, X_test, y_test = trainer.create_feature_target_datasets()
    assert X_train is not None
    assert y_train is not None
    assert X_test is not None
    assert y_test is not None

# Verifying that the best model can be found given the hyperparameters
def test_find_best_model():
    trainer = ModelTrainer()
    X_train, y_train, _, _ = trainer.create_feature_target_datasets()
    bst = FindBestModel()
    xgb = XGBRegressor(random_state=42)
    params = {'learning_rate': [0.01, 0.05, 0.1]}
    best_model = bst.find_best_model(xgb, params, X_train, y_train)
    assert best_model is not None

# Verifying that the model trainer works as expected when the make_prediction flag
# is set to True
def test_initiate_model_training_with_prediction():
    trainer = ModelTrainer()
    best_model, best_params, metric = trainer.initiate_model_training(save_model=False, make_prediction=True)
    assert best_model is not None
    assert best_params is not None
    assert metric is not None

# Verifying that the model trainer works as expected when the make_prediction flag
# is set to True
def test_initiate_model_training_without_prediction():
    trainer = ModelTrainer()
    with pytest.raises(ValueError):
        best_model, best_params, metric = trainer.initiate_model_training(save_model=False, make_prediction=False)
        assert best_model is not None
        assert best_params is not None
        assert metric is not None