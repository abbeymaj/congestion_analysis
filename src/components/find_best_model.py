# Import packages
import sys
import pandas as pd
from sklearn import set_config 
set_config(transform_output='pandas') 
from src.exception import CustomException 
from src.logger import logging
from src.components.config_entity import ModelTrainerConfig
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error 

# Creating a class to find the best model
class FindBestModel():
    '''
    This class contains methods to find the best model. The class contains two methods -
    a constructor and a method to find the best model.
    '''
    # Creating the constructor method for the class
    def __init__(self):
        '''
        This is the constructor method for the class. The constructor will instantiate
        the path to the transformed datasets.
        '''
        self.model_trainer_config = ModelTrainerConfig()
    
    # Creating a method to find the best model
    def find_best_model(
        self,
        estimator=None,
        params=None,
        train_set=None,
        target_set=None,
        cv=3
    ):
        '''
        This method is used to find the best model, given the hyperparameters.
        =================================================================================
        ----------------
        Parameters:
        ----------------
        params : dict - This is the dictionary containing the hyperparameters for the model.
        train_set : pandas dataframe - This is the training dataset.
        target_set : pandas dataframe - This is the target dataset.
        cv : int - This is the number of cross-validation folds.
        
        ----------------
        Returns:
        ----------------
        best_model : xgboost model - This is the best model.
        ==================================================================================
        '''
        try:
            # Instantiating a Grid Search object
            grid_search = GridSearchCV(
                estimator=estimator,
                param_grid=params,
                cv=cv,
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            # Fitting the train and target set to the grid search object
            grid_search.fit(train_set, target_set)
            
            # Extracting the best model 
            best_model = grid_search.best_estimator_
            
            # Extracting the best parameters
            best_params = grid_search.best_params_
            
            return (
                best_model,
                best_params
            )
        
        except Exception as e:
            raise CustomException(e, sys)
        
        