# Importing packages
import os
import sys
import pandas as pd
import sklearn
from sklearn import set_config
set_config(transform_output='pandas')
from src.exception import CustomException
from src.logger import logging
from src.components.config_entity import DataIngestionConfig
from sklearn.model_selection import train_test_split


# Creating a class to ingest the raw data from source
class DataIngestion():
    '''
    The DataIngestion class is responsible for reading the raw data at source, splitting the data
    into a train and test set, and then saving the train and test set in an "artifacts" folder.
    The class contains two methods - A constructor and a method to initiate the data ingestion
    process.
    '''
    def __init__(self):
        '''
        This is the constructor method for the DataIngestion class.
        '''
        self.ingestion_config = DataIngestionConfig()
        
    # Creating a method to initiate the data ingestion process
    def initiate_data_ingestion(self):
        '''
        This method will ingest the data from source, split the dataset into a train
        and test dataset. The function will also create the artifacts folder and store
        the train and test dataset in the artifacts folder.
        ====================================================================================
        ---------------
        Returns:
        ---------------
        train file path : str - This is the path to the train dataset.
        test file path : str - This is the path to the test dataset.
        ====================================================================================
        '''
        try:
            logging.info('Starting the data ingestion process.')
            
            # Creating the artifacts folder if it does not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            
            # Defining the data path
            data_path = 'https://github.com/abbeymaj80/my-ml-datasets/raw/refs/heads/master/project_datasets/congestion/train.parquet'
            
            # Reading the data from the source
            df = pd.read_parquet(data_path)
            
            # Dropping the "row_Id" column from the dataset
            df.drop(columns=['row_id'], axis=1, inplace=True)
                      
            # Splitting the data into a train and test set
            train_data, test_data = train_test_split(df, test_size=0.3, random_state=42)
            
            # Saving the train and test data into the artifacts folder
            train_data.to_parquet(self.ingestion_config.train_data_path, index=False, compression='gzip')
            test_data.to_parquet(self.ingestion_config.test_data_path, index=False, compression='gzip')
            
            logging.info('Data ingestion process completed.')
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        
        except Exception as e:
            raise CustomException(e, sys)
