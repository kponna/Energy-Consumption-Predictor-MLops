import os
  
"""
Defining common constant variables for training pipeline
"""
TARGET_COLUMN = "value"
PIPELINE_NAME: str = "EnergyGeneration"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "EnergyData.csv"

TRAIN_FILE_NAME: str =  "train.csv"
TEST_FILE_NAME: str = "test.csv"
VALIDATION_FILE_NAME:str = "val.csv"

TRAIN_FEATURE_FILE_NAME:str = "X_train.np"
TEST_FEATURE_FILE_NAME:str = "X_test.np"
VAL_FEATURE_FILE_NAME: str = "X_val.np"
TRAIN_TARGET_FILE_NAME:str = "y_train.np"
TEST_TARGET_FILE_NAME:str = "y_test.np"
VAL_TARGET_FILE_NAME:str = "y_val.np"
 
MODEL_FILE_NAME = "model.keras"
SAVED_MODEL_DIR = os.path.join("saved_models") 
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "energy_generation"
DATA_INGESTION_DATABASE_NAME: str = "eia_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested" 
TRAIN_VAL_TEST_SPLIT_RATIO: float = 0.7  # 70% of data for training
VALIDATION_SPLIT_RATIO: float = 0.4  # 40% of the remaining data for validation

"""
Data Validation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_VALIDATION_DIR_NAME:str= "data_validation"
DATA_VALIDATION_VALID_DIR:str = "valid"
DATA_VALIDATION_INVALID_DIR:str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR:str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME:str = 'report.yaml'

"""
Data Transformation related constant start with DATA_VALIDATION VAR NAME
"""
DATA_TRANSFORMATION_DIR:str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR:str = "transformed" 
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR:str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME:str ="scaler.pkl" 
TIME_STEPS: int = 3
PATIENCE:int = 10

"""
Model Trainer related constant start with MODEL TRAINER VAR NAME
"""
LEARNING_RATE:float = 0.0001
EPOCHS:int = 50
MODEL_TRAINER_DIR_NAME: str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR: str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME: str = "model.pkl"

TRAINING_BUCKET_NAME:str = "energygeneration"