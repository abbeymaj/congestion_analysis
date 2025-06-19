# Importing packages
import sys
import pandas as pd
from src.utils import get_current_time
from src.exception import CustomException

# Creating a class to convert user entered data into a pandas dataframe.
class CreateCustomData():
    '''
    This class is responsible for converting the user entered data into a pandas
    dataframe. The closs contains two methods - a constructor and a method to 
    convert user entered data into a pandas dataframe. 
    The objective of the class is to transform the user entered data so that the
    data can be used to make predictions.
    '''
    # Creating the constructor for the class
    def __init__(
        self,
        x:int,
        y:int,
        direction:str
    ):
        '''
        This is the constructor for the custom data class.
        '''
        self.x = x
        self.y = y
        self.direction = direction 
    
    # Creating a method to convert the user entered data into a pandas dataframe
    def create_dataframe(self):
        '''
        This method takes the data input by the user and returns a dataframe. The method 
        converts the data, input by the user on the website, into a dictionary and then creates
        a pandas dataframe using the dictionary.
        ========================================================================================
        -----------------------
        Returns:
        -----------------------
        df : pandas dataframe - A pandas dataframe of the data entered by the user.
        ========================================================================================
        '''
        try:
            # Getting the time the data was entered by the user
            current_time = get_current_time()
            
            # Creating a dictionary for the data
            data = {
                'time': [current_time],
                'x': [self.x],
                'y': [self.y],
                'direction': [self.direction]
            }
            
            # Creating a pandas dataframe from the dictionary
            df = pd.DataFrame(data)
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys)