# Importing packages
import sys
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engine.encoding import MeanEncoder


# Creating a class to mean encode the categorical features
class MeanEncode(BaseEstimator, TransformerMixin):
    '''
    This class is used to mean encode high cardinality categorical features.
    The class inherits from the BaseEstimator and TransfomerMixin classes
    from the sklearn library. The class has two methods - fit and transform.
    '''
    # Creating the constructor method for the class
    def __init__(self, cols=None):
        '''
        This is the constructor method for the MeanEncode class.
        This method takes a list of columns as inputs.
        '''
        self.cols = cols
    
    # Creating the fit method for the class
    def fit(self, X, y):
        '''
        This method uses the feature and the target set to fit the identified categorical 
        columns with the calculated mean per categorical variable.
        ========================================================================================
        ---------------------
        Parameters:
        ---------------------
        X : This is the feature dataset containing the categorical variables.
        y : This is the target dataset.
        
        ---------------------
        Returns:
        ---------------------
        The dataset after being fit with the data.
        =========================================================================================
        '''
        self.mean_encoder = MeanEncoder(variables=self.cols)
        self.mean_encoder.fit(X, y)
        return self
    
    # Creating the transform method for the class
    def transform(self, X, y=None):
        '''
        This method uses the fitted mean encoder to transform the categorical variables.
        ========================================================================================
        ---------------------
        Parameters:
        ---------------------
        X : This is the feature dataset containing the categorical variables.
        
        ---------------------
        Returns:
        ---------------------
        The dataset after being transformed.
        =========================================================================================
        '''
        if y is not None:
            return self.mean_encoder.transform(X)
        else:
            return self.mean_encoder.transform(X)
    
    
# Creating a function to generate an hour feature
def create_hour_feature(df):
    '''
    This function creates a new feature called 'hour' from the 'time' feature.
    ==============================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pd.DataFrame - This is the original dataset.
    
    ---------------------
    Returns:
    ---------------------
    df : pd.DataFrame - This is the dataset with the hour feature.
    ===============================================================================================
    '''
    try:
        df.loc[:, 'hour'] = df.loc[:, 'time'].dt.hour
        df['hour_sin'] =  np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


# Creating a function to generate an AM/PM feature
def create_am_pm_feature(df):
    '''
    This function creates a new feature called 'AM_PM' from the 'hour' feature. The "hour" feature
    itself was created from the "time" feature.
    ==============================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pd.DataFrame - This is the original dataset.
    
    ---------------------
    Returns:
    ---------------------
    df : pd.DataFrame - This is the dataset with the "AM_PM" feature.
    ===============================================================================================
    '''
    try:
        condition = [
            (df['hour'] >= 12) & (df['hour'] <= 23)
        ]
        selection = ['PM']
        df.loc[:, 'am_pm'] = np.select(condition, selection, default='AM')
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


# Creating a function to generate an "is_weekend" feature
def create_is_weekend_feature(df):
    '''
    This function creates a new feature called 'is_weekend' from the 'time' feature.
    ==============================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pd.DataFrame - This is the original dataset.
    
    ---------------------
    Returns:
    ---------------------
    df : pd.DataFrame - This is the dataset with the "AM_PM" feature.
    ===============================================================================================
    '''
    try:
        df.loc[:, 'is_weekend'] = df.loc[:, 'time'].dt.dayofweek > 4
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


# Creating a function to create the direction feature
def create_direction_feature(df):
    '''
    This function creates a new feature called 'direction'. The feature is created from the 'x', 
    'y' and 'direction' features. 
    ==============================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pd.DataFrame - This is the original dataset.
    
    ---------------------
    Returns:
    ---------------------
    df : pd.DataFrame - This is the dataset with the "AM_PM" feature.
    ===============================================================================================
    '''
    try:
        df.loc[:, 'x_y_direction'] = df['x'].astype(str) + '_' + df['y'].astype(str) + '_' + df['direction']
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


# Creating a function to drop the non-essential features
def drop_non_essential_features(df):
    '''
    This function drops the non-essential features from the dataset. The features that are
    dropped are 'time', 'x', 'y' and 'direction'.
    ==============================================================================================
    ---------------------
    Parameters:
    ---------------------
    df : pd.DataFrame - This is the original dataset.
    
    ---------------------
    Returns:
    ---------------------
    df : pd.DataFrame - This is the dataset with the "AM_PM" feature.
    ===============================================================================================
    '''
    try:
        df.drop(labels=['time', 'hour', 'x', 'y', 'direction'], axis=1, inplace=True)
        return df
    
    except Exception as e:
        raise CustomException(e, sys)