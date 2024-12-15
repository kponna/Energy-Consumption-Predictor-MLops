import sys
import os
import numpy as np
import pandas as pd

"""
Defining common constant variables for training pipeline
"""

TARGET_COLUMN = "value"
PIPELINE_NAME: str = "EnergyGeneration"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "EnergyData.csv"

# TRAIN_FILE_NAME: str = "train.csv"
# TEST_FILE_NAME: str = "test.csv"

# PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"
# MODEL_FILE_NAME = "model.pkl"
# SCHEMA_FILE_PATH = os.path.join("data_schema", "schema.yaml")
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
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.2 