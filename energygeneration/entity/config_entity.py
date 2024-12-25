from datetime import datetime
import os
from energygeneration.constant import *

class TrainingPipelineConfig:
    def __init__(self, timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = PIPELINE_NAME
        self.artifact_dir = os.path.join(ARTIFACT_DIR, timestamp)
        self.timestamp: str = timestamp


class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir: str = os.path.join(training_pipeline_config.artifact_dir, DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_FEATURE_STORE_DIR, FILE_NAME)
        self.training_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TRAIN_FILE_NAME)
        self.testing_file_path: str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, TEST_FILE_NAME)
        self.validation_file_path :str = os.path.join(self.data_ingestion_dir, DATA_INGESTION_INGESTED_DIR, VALIDATION_FILE_NAME)
        self.train_val_test_split_ratio: float = TRAIN_VAL_TEST_SPLIT_RATIO
        self.window_size:int = WINDOW_SIZE
        self.validation_split_ratio = VALIDATION_SPLIT_RATIO
        self.collection_name: str = DATA_INGESTION_COLLECTION_NAME
        self.database_name: str = DATA_INGESTION_DATABASE_NAME

class DataValidationConfig:
    def __init__(self,training_pipeline_config: TrainingPipelineConfig):
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
        
