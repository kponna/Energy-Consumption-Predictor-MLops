import os
from datetime import datetime
from energygeneration.constant import *

class TrainingPipelineConfig:
    """
    Configuration for the training pipeline, including artifact directory and timestamp.
    """
    def __init__(self, timestamp=datetime.now()):
        """
        Initialize the configuration with a unique timestamp.
        """
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = PIPELINE_NAME
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)
        self.model_dir=os.path.join("final_model")
        self.timestamp: str = timestamp


class DataIngestionConfig:
    """
    Configuration for data ingestion, including paths for feature store, train, test, and validation files.
    """
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data ingestion paths and parameters.
        """
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        self.validation_file_path :str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, VALIDATION_FILE_NAME)
        self.train_val_test_split_ratio: float = TRAIN_VAL_TEST_SPLIT_RATIO
        self.time_steps:int = TIME_STEPS
        self.validation_split_ratio = VALIDATION_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = DATA_INGESTION_DATABASE_NAME

class DataValidationConfig:
    """
    Configuration for data validation, including directories for valid and invalid data, and drift report path.
    """
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
        """
        Initialize data validation paths.
        """
        self.data_validaton_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
        self.valid_data_dir:str = os.path.join(self.data_validaton_dir,DATA_VALIDATION_VALID_DIR)
        self.invalid_data_dir:str = os.path.join(self.data_validaton_dir,DATA_VALIDATION_INVALID_DIR)
        self.valid_train_file_path:str = os.path.join(self.valid_data_dir,TRAIN_FILE_NAME)
        self.valid_test_file_path:str = os.path.join(self.valid_data_dir,TEST_FILE_NAME)
        self.valid_val_file_path:str = os.path.join(self.valid_data_dir,VALIDATION_FILE_NAME)
        self.invalid_train_file_path:str = os.path.join(self.invalid_data_dir,TRAIN_FILE_NAME)
        self.invalid_test_file_path:str = os.path.join(self.invalid_data_dir,TEST_FILE_NAME)
        self.invalid_val_file_path:str = os.path.join(self.invalid_data_dir,VALIDATION_FILE_NAME)
        self.drift_report_file_path:str = os.path.join(self.data_validaton_dir,DATA_VALIDATION_DRIFT_REPORT_DIR,DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)
        
class DataTransformationConfig:
    """
    Configuration for data transformation, including paths for transformed data and preprocessing objects.
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        """
        Initialize data transformation paths.
        """
        self.data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR)
        self.transformed_X_train_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_FEATURE_FILE_NAME )
        self.transformed_X_test_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FEATURE_FILE_NAME )
        self.transformed_X_val_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,VAL_FEATURE_FILE_NAME )
        self.transformed_y_train_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_TARGET_FILE_NAME )
        self.transformed_y_test_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_TARGET_FILE_NAME )
        self.transformed_y_val_file_path:str = os.path.join(self.data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,VAL_TARGET_FILE_NAME )
        self.transformed_object_file_path: str = os.path.join( self.data_transformation_dir, DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCESSING_OBJECT_FILE_NAME,)

class ModelTrainerConfig:
    """
    Configuration for model training, including paths for trained models.
    """
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        """
        Initialize model trainer paths.
        """
        self.model_trainer_dir:str = os.path.join(training_pipeline_config.artifact_dir,MODEL_TRAINER_DIR_NAME)
        self.trained_model_file_path:str = os.path.join(self.model_trainer_dir,MODEL_TRAINER_TRAINED_MODEL_DIR,MODEL_FILE_NAME) 