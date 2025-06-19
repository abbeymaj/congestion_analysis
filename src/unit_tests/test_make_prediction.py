# Import packages
import os
import pytest
import pandas as pd
from src.components.create_custom_data import CreateCustomData

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