# Importing packages
import dagshub
import mlflow
from mlflow import MlflowClient
from src.utils import save_run_params
from src.components.model_trainer import ModelTrainer

# Running the training script
if __name__ == '__main__':
    
    # Initiating the dagshub client
    dagshub.init(repo_owner='abbeymaj', repo_name='congestion_analysis', mlflow=True)
    
    # Setting the tracking URI for the model
    model_uri = 'https://dagshub.com/abbeymaj/congestion_analysis.mlflow'
    mlflow.set_tracking_uri(model_uri)
    
    # Instantiating the mlflow client
    client = MlflowClient()
    
    # Creating the experiment
    experiment_id = client.create_experiment('ca_training_1')
    
    # Starting the training run
    run_params = {}
    with mlflow.start_run(run_name='training_pipeline_1', experiment_id=experiment_id) as run:
        # Fetch the run id
        run_id = run.info.run_id
        # Instantiating the model trainer
        trainer = ModelTrainer()
        # Fetching the best model and best parameters 
        best_model, best_params = trainer.initiate_model_training(save_model=False, make_prediction=False)
        # Logging the best model and the best parameters
        mlflow.log_params(best_params)
        model_info = mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path='models/training_model_1',
            registered_model_name='training_model_1'
        )
        
        # Fetch the latest version of the model and the model name
        latest_version_info = client.get_latest_versions('training_model_1', stages=['None'])[0]
        model_name = latest_version_info.name
        latest_version = latest_version_info.version
        
        # Storing the model uri and run id into a dictionary
        run_params['model_uri'] = model_info.model_uri
        run_params['run_id'] = run_id
        run_params['model_name'] = model_name
        run_params['latest_version'] = latest_version
    
    # Saving the run parameters as a json file
    save_run_params(run_params)
        