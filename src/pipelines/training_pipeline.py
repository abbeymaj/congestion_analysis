# Importing packages
import dagshub
import mlflow
from mlflow import MlflowClient
from src.components.model_trainer import ModelTrainer

# Running the training script
if __name__ == '__main__':
    
    # Initiating the dagshub client
    dagshub.init(repo_owner='abbeymaj', repo_name='congestion_analysis', mlflow=True)
    
    # Setting the tracking URI for the model
    model_uri = 'https://dagshub.com/abbeymaj/congestion_analysis.mlflow'
    mlflow.set_tracking_uri(model_uri)