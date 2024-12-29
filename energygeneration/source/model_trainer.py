import os
import sys 
from energygeneration.constant import TIME_STEPS
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging 
from energygeneration.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from energygeneration.entity.config_entity import ModelTrainerConfig 
from energygeneration.utils.main_utils.utils import save_object,load_numpy_array_data,load_object 
from energygeneration.utils.ml_utils.metric.reg_metric import get_model_score
from energygeneration.utils.ml_utils.model.estimator import EnergyLstmModel 
from mlflow.models.signature import infer_signature
from tensorflow.keras.models import Model
import mlflow 
import dagshub
dagshub.init(repo_owner='kponna', repo_name='Energy-Generation-Predictor-MLops', mlflow=True)

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise EnergyGenerationException(e, sys) 


    def track_mlflow(self, best_model, train_metrics, val_metrics, test_metrics, X_train_sample, scaler):
        with mlflow.start_run(run_name="Model_Training_And_Evaluation"):
            # Log train metrics
            mlflow.log_metric("train_mae", train_metrics.mae)
            mlflow.log_metric("train_mse", train_metrics.mse)
            mlflow.log_metric("train_rmse", train_metrics.rmse)
            mlflow.log_metric("train_r2", train_metrics.r2)

            # Log validation metrics
            mlflow.log_metric("val_mae", val_metrics.mae)
            mlflow.log_metric("val_mse", val_metrics.mse)
            mlflow.log_metric("val_rmse", val_metrics.rmse)
            mlflow.log_metric("val_r2", val_metrics.r2)

            # Log test metrics
            mlflow.log_metric("test_mae", test_metrics.mae)
            mlflow.log_metric("test_mse", test_metrics.mse)
            mlflow.log_metric("test_rmse", test_metrics.rmse)
            mlflow.log_metric("test_r2", test_metrics.r2)

            # Log the trained model
            input_example = X_train_sample[:1]
            signature = infer_signature(input_example, best_model.predict(input_example))
            # mlflow.tensorflow.log_model(best_model, "Trained_LSTM_Model", input_example=input_example, signature=signature)
            mlflow.tensorflow.log_model(best_model.model, "Trained_LSTM_Model", input_example=input_example, signature=signature)
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Loading transformed data...")
            X_train = load_numpy_array_data(self.data_transformation_artifact.transformed_X_train_file_path)
            y_train = load_numpy_array_data(self.data_transformation_artifact.transformed_y_train_file_path)
            X_val = load_numpy_array_data(self.data_transformation_artifact.transformed_X_val_file_path)
            y_val = load_numpy_array_data(self.data_transformation_artifact.transformed_y_val_file_path)
            X_test = load_numpy_array_data(self.data_transformation_artifact.transformed_X_test_file_path)
            y_test = load_numpy_array_data(self.data_transformation_artifact.transformed_y_test_file_path)

            logging.info(f"Data shapes - X_train: {X_train.shape}, y_train: {y_train.shape}, "
                         f"X_val: {X_val.shape}, y_val: {y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}")

            # Build and compile the LSTM model
            logging.info("Building the LSTM model...") 
            # Ensure you have the scaler loaded from your transformation step
            scaler = load_object(self.data_transformation_artifact.transformed_object_file_path) 
            Energy_Model = EnergyLstmModel(input_shape=(TIME_STEPS,11),
                    model_checkpoint_path=self.model_trainer_config.trained_model_file_path)
            Energy_Model.build_model() 
            logging.info("Starting model training...")
            Energy_Model.train_model(X_train, y_train, X_val, y_val, epochs=50) 
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True) 
            save_object(self.model_trainer_config.trained_model_file_path, obj=Energy_Model)

            # Evaluate and predict
            logging.info("Making Predictions with the model...")
            y_pred_test = Energy_Model.predict(X_test)
            y_pred_train = Energy_Model.predict(X_train)
            y_pred_val = Energy_Model.predict(X_val)
            # Log the shapes of the predicted values
            logging.info(f"Predictions shapes - y_pred_test: {y_pred_test.shape}, y_pred_train: {y_pred_train.shape}, y_pred_val: {y_pred_val.shape}")
             
            y_test_rescaled = scaler.inverse_transform(y_test).flatten()
            y_pred_test_rescaled = scaler.inverse_transform(y_pred_test.reshape(-1, 1)).flatten()
 
            y_train_rescaled = scaler.inverse_transform(y_train).flatten()
            y_pred_train_rescaled = scaler.inverse_transform(y_pred_train.reshape(-1, 1)).flatten()
 
            y_val_rescaled = scaler.inverse_transform(y_val).flatten()
            y_pred_val_rescaled = scaler.inverse_transform(y_pred_val.reshape(-1, 1)).flatten() 
            logging.info(f"Predictions shapes - y_pred_test_rescaled: {y_pred_test_rescaled.shape}, y_pred_train_rescaled: {y_pred_train_rescaled.shape}, y_pred_val_rescaled: {y_pred_val_rescaled.shape}")
            logging.info(f"Predictions shapes - y_test_rescaled: {y_test_rescaled.shape}, y_train_rescaled: {y_train_rescaled.shape}, y_val_rescaled: {y_val_rescaled.shape}")
            # Log the first few values for cross-checking
            logging.info(f"First 5 rescaled predictions and true values:")
            logging.info(f"y_pred_test_rescaled[:5]: {y_pred_test_rescaled[:5]}")
            logging.info(f"y_test_rescaled[:5]: {y_test_rescaled[:5]}")
            logging.info(f"y_pred_train_rescaled[:5]: {y_pred_train_rescaled[:5]}")
            logging.info(f"y_train_rescaled[:5]: {y_train_rescaled[:5]}")
            logging.info(f"y_pred_val_rescaled[:5]: {y_pred_val_rescaled[:5]}")
            logging.info(f"y_val_rescaled[:5]: {y_val_rescaled[:5]}")
            # Calculate regression metrics on training, validation, and test sets
            logging.info("Calculating model evaluation metrics...")
            regression_train_metric = get_model_score(y_train_rescaled, y_pred_train_rescaled)
            regression_test_metric = get_model_score(y_test_rescaled,y_pred_test_rescaled)
            regression_val_metric = get_model_score(y_val_rescaled,y_pred_val_rescaled) 
            logging.info(f"Model Test evaluation metrics: {regression_test_metric}")
            logging.info(f"Model Train evaluation metrics: {regression_train_metric}")
            logging.info(f"Model Val evaluation metrics: {regression_val_metric}")
 
            # Track the experiment in MLflow
            self.track_mlflow(
                best_model=Energy_Model,
                train_metrics=regression_train_metric,
                val_metrics=regression_val_metric,
                test_metrics=regression_test_metric,
                X_train_sample=X_train,
                scaler=scaler,
            ) 

            # Launch the MLflow UI
            # subprocess.run(["mlflow", "ui"])
            save_object("final_model/model.keras",Energy_Model) 

            # print(mlflow ui)
            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric,
                val_metric_artifact=regression_val_metric
            ) 
            logging.info(f"Model Trainer artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise EnergyGenerationException(e, sys)
         