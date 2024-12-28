import os
import sys 
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging 
from energygeneration.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
from energygeneration.entity.config_entity import ModelTrainerConfig 
from energygeneration.utils.main_utils.utils import save_object,load_numpy_array_data,load_object 
from energygeneration.utils.ml_utils.metric.reg_metric import get_model_score
from energygeneration.utils.ml_utils.model.estimator import EnergyLstmModel 

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise EnergyGenerationException(e, sys)

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
            estimator = EnergyLstmModel(input_shape=(12,11),
                    model_checkpoint_path=self.model_trainer_config.trained_model_file_path)
            estimator.build_model() 
            logging.info("Starting model training...")
            estimator.train_model(X_train, y_train, X_val, y_val, epochs=50) 
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True) 
            save_object(self.model_trainer_config.trained_model_file_path, obj=estimator)

            # Evaluate and predict
            logging.info("Evaluating the model...")
            y_pred_test = estimator.predict(X_test)
            y_pred_train = estimator.predict(X_train)
            y_pred_val = estimator.predict(X_val)
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
            logging.info("Calculating model metrics...")
            regression_train_metric = get_model_score(y_train_rescaled, y_pred_train_rescaled)
            regression_test_metric = get_model_score(y_test_rescaled,y_pred_test_rescaled)
            regression_val_metric = get_model_score(y_val_rescaled,y_pred_val_rescaled) 
            logging.info(f"Model Test evaluation metrics: {regression_test_metric}")
            logging.info(f"Model Train evaluation metrics: {regression_train_metric}")
            logging.info(f"Model Val evaluation metrics: {regression_val_metric}")

            model_trainer_artifact = ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=regression_train_metric,
                test_metric_artifact=regression_test_metric,
                val_metric_artifact=regression_val_metric
            ) 
            return model_trainer_artifact
        except Exception as e:
            raise EnergyGenerationException(e, sys)
         