import sys
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
 
# MODEL_FILE_NAME = "model.pkl"
 
SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
# SCHEMA_DROP_COLS = ["respondent-name", "type-name"]

# SAVED_MODEL_DIR = os.path.join("saved_models")

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "energy_generation"
DATA_INGESTION_DATABASE_NAME: str = "eia_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
##### DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2

# Constants for time series modeling

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
TIME_STEPS: int = 12   
## KNN imputer to replace nan values
# DATA_TRANSFORMATION_IMPUTER_PARAMS:dict = {
#     "missing_values":np.nan,
#     "n_neighbors": 3,
#     "weights": "uniform",
# } 