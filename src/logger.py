# Importing packages
import logging
import os
from datetime import datetime

# Creating the log file
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Creating the directory in which the log file will be stored
logs_path = os.path.join(os.getcwd(), 'logs')
os.makedirs(logs_path, exist_ok=True)

# Creating the full path to store the log files
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Overriding the basicConfig of the logging module
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)