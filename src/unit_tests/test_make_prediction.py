# Import packages
import os
import pytest
import pandas as pd
from src.components.create_custom_data import CreateCustomData
from src.components.make_predictions import MakePredictions

# Verifying that the create_dataframe method works as expected
def test_create_dataframe():
    custom_data = CreateCustomData(1, 2, 'N')
    df = custom_data.create_dataframe()
    assert df is not None
    assert isinstance(df, pd.DataFrame)
    assert 'time' in list(df.columns)
    assert 'x' in list(df.columns)
    assert 'y' in list(df.columns)
    assert 'direction' in list(df.columns)
    assert len(df) == 1

# Verifying that the system can retrieve the model parameters
def test_retrieve_model_params():
    prediction_class = MakePredictions()
    runs_data, latest_model_uri = prediction_class.retrieve_model_params()
    assert runs_data is not None
    assert latest_model_uri is not None

# Verifying that the system can retrive the model from the model registry
def test_retrieve_model():
    prediction_class = MakePredictions()
    model = prediction_class.retrieve_model()
    assert model is not None

# Verifying that the transformed data is a pandas dataframe
def test_transformed_data_type():
    custom_data = CreateCustomData(1, 2, 'N')
    df = custom_data.create_dataframe()
    prediction_class = MakePredictions()
    transformed_data = prediction_class.predict(df, test_transformed_features=True)
    assert transformed_data is not None
    assert isinstance(transformed_data, pd.DataFrame)

# Verifying that the predictions can be generated on a datapoint
def test_make_prediction():
    custom_data = CreateCustomData(1, 2, 'N')
    df = custom_data.create_dataframe()
    prediction_class = MakePredictions()
    preds = prediction_class.predict(df)
    assert preds is not None
    