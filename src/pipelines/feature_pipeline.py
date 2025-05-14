# Importing packages
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.config_entity import DataIngestionConfig

# Running the feature creation pipeline
if __name__ == '__main__':
    
    # Creating the artifacts folder and ingesting the data
    ingestion_obj = DataIngestion()
    path_to_train_data, path_to_test_data = ingestion_obj.initiate_data_ingestion()h
    
    # Transforming the datasets
    transformation_obj = DataTransformation()
    train_set, test_set = transformation_obj.initiate_data_transformation(path_to_train_data, path_to_test_data)