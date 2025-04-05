# Importing packages
import sys

# Creating a function to fetch error messages from the sys module
def fetch_error_message(error, error_detail:sys):
    '''
    This function fetches the error information from the sys module.
    ====================================================================
    ------------------
    Parameters:
    ------------------
    error - str : This is the error message.
    error_detail: This is the error detail from the sys module.
    
    ------------------
    Returns:
    ------------------
    error_message - This is the error message from the sys module.
    =====================================================================
    '''
    _, _, exc_tbl = error_detail.exc_info()
    file_name = exc_tbl.tb_frame.f_code.co_filename
    line_number = exc_tbl.tb_lineno
    error_message = f"Error occured in script name [{0}] at line number [{1}] with error message [{2}]".format(
        file_name, line_number, str(error)
    )
    return error_message


# Creating a custom exception class to display exceptions raised by the system
class CustomException(Exception):
    '''
    The CustomException class is a custom class to display system exceptions. 
    This class inherits from the Python Exception class.
    The class contains two methods - The constructor and a method to display the error message.
    '''
    # Defining the class constructor
    def __init__(self, error_message, error_detail:sys):
        '''
        This is the constructor method of the CustomException class.
        Ths constructor takes two parameters - error_message and error_detail.
        '''
        # Instantiating the parent class and passing the error message to it.
        super().__init__(error_message)
        self.error_message = fetch_error_message(error_message, error_detail=error_detail)
        
    # Creating a method to display the error message
    def __str__(self):
        return self.error_message