# Importing packages
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.store_features import FeatureStoreCreation

# Running the feature creation pipeline
if __name__ == '__main__':
    
    # Creating the artifacts folder and ingesting the data
    ingestion_obj = DataIngestion()
    path_to_train_data, path_to_test_data = ingestion_obj.initiate_data_ingestion()
    
    # Transforming the datasets
    transformation_obj = DataTransformation()
    train_dataset, test_dataset = transformation_obj.initiate_data_transformation(path_to_train_data, path_to_test_data)
        
    # Storing the transformed datasets
    feature_store_obj = FeatureStoreCreation()
    feature_store_obj.create_feature_store(train_dataset, test_dataset)