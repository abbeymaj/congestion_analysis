# Importing packages
import sys
import joblib
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime
from sklearn import set_config
set_config(transform_output='pandas')
from feature_engine.encoding import MeanEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from src.logger import logging
from src.exception import CustomException
from src.utils import create_hour_feature
from src.utils import create_am_pm_feature
from src.utils import create_is_weekend_feature
from src.utils import create_direction_feature
from src.utils import drop_non_essential_features
from src.components.config_entity import DataIngestionConfig
from src.components.config_entity import DataTransformationConfig


# Creating a class to transform the data
class DataTransformation():
    '''
    The DataTransformation class is responsible for transforming the train and test datasets. The 
    class is also responsible for creating the preprocessor object and saving it to the artifacts
    folder. 
    '''
    # Creating the constructor method for the DataTransformation class
    def __init__(self):
        '''
        This is the constructor method for the DataTransformation class. This method instantiates
        the path where the train and test datasets are stored as well as the path where the 
        preprocessor object will be stored.
        '''
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
    
    # Creating a method to generate new features and drop non-essential features
    def generate_features(self, df:pd.DataFrame)->pd.DataFrame:
        '''
        This method generates new features and drops non-essential features from the
        dataset.
        ==============================================================================
        ----------------
        Parameters:
        ----------------
        df : pd.DataFrame - The original dataset that is to be transformed.
        
        ----------------
        Returns:
        ----------------
        df : pd.DataFrame - The transformed dataset with new features.
        '''
        try:
            # Checking if 'time' is in the datetime format and, if not, then
            #converting it to the datetime format.
            if not is_datetime(df['time']):
                df['time'] = pd.to_datetime(df['time'])
            
            # Creating new features in the dataset
            df = create_hour_feature(df)
            df = create_am_pm_feature(df)
            df = create_is_weekend_feature(df)
            df = create_direction_feature(df)
            
            # Dropping the non-essential features from the dataset
            df = drop_non_essential_features(df)
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)
    
    
    # Creating a method to create the preprocessor object
    def create_preprocessor_obj(self):
        '''
        This method creates the preprocessor object.
        =============================================================================
        ------------------
        Returns:
        ------------------
        preprocessor : pipeline object - Returns the preprocessor object. 
        =============================================================================
        '''
        try:
            logging.info('Creating the preprocessor object.')
            
            # Listing the feature for mean encoding
            mean_encoding_features = ['x_y_direction']
            # Listing the features for one hot encoding
            ohe_features = ['am_pm', 'is_weekend']
            
            # Creating the one hot encoder pipeline
            ohe_pipeline = Pipeline(
                steps=[
                    ('ohe', OneHotEncoder(sparse_output=False))
                ]
            )
            
            # Creating the mean encoder pipeline
            me_pipeline = Pipeline(
                steps=[
                    ('mean_encoder', MeanEncoder())
                ]
            )
            
            # Combining the pipelines into a column transformer
            preprocessor = ColumnTransformer(
                [
                    ('ohe_pipeline', ohe_pipeline, ohe_features),
                    ('me_pipeline', me_pipeline, mean_encoding_features)
                ], remainder='passthrough'
            )
            
            logging.info('Preprocessor object created.')
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating a method to initiate the data transformation process
    def initiate_data_transformation(self, train_data_path:str, test_data_path:str, save_object=True):
        '''
        This method performs the data transformation on the feature set.
        ===============================================================================
        ----------------
        Parameters:
        ----------------
        train_path : str - The path in which the training data is stored.
        test_path : str - The path in which the test data is stored.
        
        ----------------
        Returns:
        ----------------
        train_set : parquet file - The file used for training the model.
        test_set : parquet file - The array used for testing the model.
        preprocessor object path : str - The path in which the preprocessor object 
        is stored.
        ================================================================================
        '''
        try:
            logging.info('Starting the data transformation process.')
            
            # Reading the train and test datasets
            train_data = pd.read_parquet(self.data_ingestion_config.train_data_path)
            test_data = pd.read_parquet(self.data_ingestion_config.test_data_path)
            
            # Conducting feature engineering on the train and test data and dropping
            # any features that are not required
            train_df = self.generate_features(train_data)
            test_df = self.generate_features(test_data)
            
            # Create the preprocessor object
            preprocessor = self.create_preprocessor_obj()
            
            # Separating the train data into the feature and target set
            train_features = train_df.copy().drop(labels=['congestion'], axis=1)
            train_target = train_df['congestion'].copy()
            
            # Separating the test data into the feature and target set
            test_features = test_df.copy().drop(labels=['congestion'], axis=1)
            test_target = test_df['congestion'].copy()
            
            # Transforming the train and test datasets
            train_input_features = preprocessor.fit_transform(train_features, train_target)
            test_input_features = preprocessor.transform(test_features)
            
            # Concatenating the train and test sets
            train_data_combined = pd.concat([train_input_features, train_target], axis=1)
            test_data_combined = pd.concat([test_input_features, test_target], axis=1)
            
            # Saving the preprocessor object
            if save_object:
                joblib.dump(preprocessor, self.data_transformation_config.preprocessor_obj_path)
            
            logging.info('Data transformation process completed and preprocessor object saved.')
            
            return (
                train_data_combined,
                test_data_combined
            )  
            
        except Exception as e:
            raise CustomException(e, sys)