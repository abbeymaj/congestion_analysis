# Importing packages
import sys
import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.config_entity import StoreFeatureConfig

# Creating a class to store the transformed datasets.
class FeatureStoreCreation():
    '''
    This class contains methods to store the transformed datasets. The class
    contains two methods - a constructor and a method to store the transformed
    datasets.
    '''
    # Creating the class constructor
    def __init__(self):
        '''
        This is the constructor of the FeatureStoreCreation class. It initializes
        the path in which the transformed datasets will be stored.
        '''
        self.feature_store_config = StoreFeatureConfig()
    
    # Creating the method to store the transformed datasets and also create the 
    # feature store folder.
    def create_feature_store(self, train_set, test_set):
        '''
        This method stores the transformed datasets in the feature store folder.
        ==========================================================================
        ----------------
        Parameters:
        ----------------
        train_path : pandas dataframe - This is the train dataset.
        test_path : pandas dataframe - This is the test dataset.
        
        ----------------
        Returns:
        ----------------
        transformed train data path : str - Returns the path to the transformed train dataset.
        transformed test data path : str - Returns the path to the transformed test dataset.
        ===========================================================================
        '''
        try:
            # Creating the feature store directory
            dir_name = os.path.dirname(self.feature_store_config.xform_train_path)
            os.makedirs(dir_name, exist_ok=True)
            
            # Saving the transformed datasets
            train_set.to_parquet(self.feature_store_config.xform_train_path, index=False, compression='gzip')
            test_set.to_parquet(self.feature_store_config.xform_test_path, index=False, compression='gzip')
            
            return (
                self.feature_store_config.xform_train_path,
                self.feature_store_config.xform_test_path
            )
            
        
        except Exception as e:
            raise CustomException(e, sys)