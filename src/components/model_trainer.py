# Importing packages
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn import set_config
set_config(transform_output='pandas')
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.components.config_entity import StoreFeatureConfig
from src.components.config_entity import ModelTrainerConfig
from src.components.find_best_model import FindBestModel

# Creating a class to train the model
class ModelTrainer():
    '''
    This class contains methods to train the model. The model is saved to a model
    registry using MLFlow. If needed, the model can be saved to the artifacts 
    folder. 
    '''
    # Creating the constructor method for the ModelTrainer class
    def __init__(self):
        '''
        This is the constructor for the ModelTrainer class. The constructor initializes
        the path to the transformed datasets and the preprocessor object.
        '''
        self.feature_store_config = StoreFeatureConfig()
        self.model_trainer_config = ModelTrainerConfig()
    
    # Creating a method to create the feature and target datasets
    def create_feature_target_datasets(self):
        '''
        This method creates the feature and target datasets.
        ============================================================================       
        -------------------
        Returns:
        -------------------
        X_train : pandas dataframe - The training feature set.
        y_train : pandas dataframe - The training target set.
        X_test : pandas dataframe - The test feature set.
        y_test : pandas dataframe - The test target set.
        =============================================================================
        '''
        try:
            # Reading the datasets from the feature store folder
            train_dataset = pd.read_parquet(self.feature_store_config.xform_train_path)
            test_dataset = pd.read_parquet(self.feature_store_config.xform_test_path)
            
            # Splitting the train set into a feature and target set
            X_train = train_dataset.copy().drop(labels=['congestion'], axis=1)
            y_train = train_dataset['congestion'].copy()
            
            # Splitting the test set into a feature and target set
            X_test = test_dataset.copy().drop(labels=['congestion'], axis=1)
            y_test = test_dataset['congestion'].copy()
            
            return (
                X_train,
                y_train,
                X_test,
                y_test
            )
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating a method to train the model
    def initiate_model_training(self, save_model=True, make_prediction=True):
        '''
        This method trains the model and then saves the trained model to the artifacts
        folder.
        ===================================================================================
        ------------------------
        Parameters:
        ------------------------
        save_model : bool - This determines if the model should be saved in the artifacts
        folder.
        
        ------------------------
        Returns:
        ------------------------
        model_path : str - This is the path to the saved model.
        best_params : dict - This is the best hyperparameters for the best model.
        metric : float - This is the metric from the prediction.
        ====================================================================================
        '''
        try:
            # Fetching the datasets
            X_train, y_train, X_test, y_test = self.create_feature_target_datasets()
            
            # Instantiating the XGBoost regressor 
            xgb = XGBRegressor(
                objective='reg:squarederror',
                booster='gbtree',
                grow_policy='depthwise', 
                random_state=42
                )
            
            # Defining the hyperparameters to tune
            params = {
                'learning_rate': [0.001, 0.01, 0.1],
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 5, 7]
            }
            
            # Finding the best model 
            bst = FindBestModel()
            best_model, best_params = bst.find_best_model(
                estimator=xgb,
                params=params,
                train_set=X_train,
                target_set=y_train
            )
            
            # Saving the model if save_model is True
            if save_model is True:
                joblib.dump(best_model, self.model_trainer_config.model_path)
            
            # Making a prediction if make_prediction is True
            if make_prediction is True:
                y_pred = best_model.predict(X_test)
                metric = np.sqrt(mean_squared_error(y_test, y_pred))
                
                return (
                    best_model,
                    best_params,
                    metric
                )
            else:
                return(
                    best_model,
                    best_params
                )
        
        except Exception as e:
            raise CustomException(e, sys)
         