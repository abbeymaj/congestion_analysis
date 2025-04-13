# Importing packages
from src.components.data_ingestion import DataIngestion

# Running the feature creation pipeline
if __name__ == '__main__':
    
    # Creating the artifacts folder and ingesting the data
    ingestion_obj = DataIngestion()
    train_data_path, test_data_path = ingestion_obj.initiate_data_ingestion()