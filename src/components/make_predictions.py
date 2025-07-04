# Importing packages
import sys
import pandas as pd
import mlflow
import dagshub
import joblib
from sklearn import set_config
set_config(transform_output='pandas')
from src.utils import load_run_params
from src.utils import read_json_file
from src.exception import CustomException
from src.components.config_entity import DataTransformationConfig
from src.components.data_transformation import DataTransformation

# Creating a class to make predictions based on the data provided by the user
class MakePredictions():
    '''
    This class is responsible for making predictions on the data received from 
    the website.
    '''
    # Creating the constructor for the class
    def __init__(self):
        '''
        This is the constructor for the MakePredictions class.
        '''
        self.preprocessor_obj_path = DataTransformationConfig()
        self.model_uri = 'https://dagshub.com/abbeymaj/congestion_analysis.mlflow'
    
    # Creating a method to retrieve the latest parameters for the trained model
    def retrieve_model_params(self):
        '''
        This method retrieves the model parameters for the latest trained model.
        ===================================================================================
        ----------------
        Returns:
        ----------------
        runs_data : json - This is the json file containing the model parameters for the 
        latest model.
        
        latest_model_uri : str - This is the uri for the latest model.
        ===================================================================================
        '''
        try:
            # Fetching the runs parameters from the latest json file
            runs_params_json = load_run_params()
            
            # Reading the json file 
            runs_data = read_json_file(runs_params_json)
            
            # Fetch the model uri for the latest model
            latest_model_uri = runs_data['model_uri']
            
            return runs_data, latest_model_uri
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating a method to fetch the model from the model registry
    def retrieve_model(self):
        '''
        This method retrieves the trained model from the model registry.
        ===================================================================================
        ----------------
        Returns:
        ----------------
        model : xgboost.core.Booster - This is the trained model from the model registry.
        ===================================================================================
        '''
        try:
            # Initializing the connection to the model registry
            dagshub.init(repo_owner='abbeymaj', repo_name='congestion_analysis', mlflow=True)
            
            # Setting the tracking uri
            mlflow.set_tracking_uri(self.model_uri)
            
            # Retrieve the latest model URI
            _, model_uri = self.retrieve_model_params()
            
            # Fetching the model from the model registry
            model = mlflow.pyfunc.load_model(model_uri)
            
            return model
        
        except Exception as e:
            raise CustomException(e, sys)
    
    # Creating a method to make predictions on the data entered by the user
    def predict(self, features, test_transformed_features=False):
        '''
        This method makes predictions using the feature inputs from the web page and 
        the trained model. This method also transforms the input data using the 
        preprocessor object before making the predictions.
        ============================================================================================
        -------------------
        Parameters:
        -------------------
        features : pandas dataframe - This is the feature data input received from the web page.
        
        -------------------
        Returns:
        -------------------
        preds : This is the prediction based on the input features.
        =============================================================================================
        '''
        try:
            # Instantiating the preprocessor object
            preprocessor = joblib.load(self.preprocessor_obj_path.preprocessor_obj_path)
            
            # Retrieving the model from the model registry
            model = self.retrieve_model()
            
            # Transforming the features using the preprocessor object
            data_transform = DataTransformation()
            feature_eng = data_transform.generate_features(features)
            transformed_features = preprocessor.transform(feature_eng)
            
            # Returning the preprocessor object for unit test or running
            # the predictions
            
            if test_transformed_features is True:
                return transformed_features
            else:
                # Making predictions using the transformed dataset
                preds = model.predict(transformed_features)
                return preds
        
        except Exception as e:
            raise CustomException(e, sys)