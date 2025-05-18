# Importing packages
import os
from dataclasses import dataclass

# Creating the config class for data ingestion
@dataclass
class DataIngestionConfig():
    '''
    This class defines the path for the train and test datasets.
    '''
    train_data_path: str = os.path.join('artifacts', 'train_data.parquet')
    test_data_path: str = os.path.join('artifacts', 'test_data.parquet')

# Creating a config class for data transformation
@dataclass
class DataTransformationConfig():
    '''
    This class defines the path in which the transformed data will be stored.
    '''
    preprocessor_obj_path: str = os.path.join('artifacts', 'preprocessor.joblib')

# Creating a config class to create the feature store to store the transformed
# datasets.
@dataclass
class StoreFeatureConfig():
    '''
    This class defines the path in which the transformed datasets will be 
    stored.
    '''
    xform_train_path: str = os.path.join('feature_store', 'xform_train_set.parquet')
    xform_test_path: str = os.path.join('feature_store', 'xform_test_set.parquet')