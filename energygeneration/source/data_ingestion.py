from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.artifact_entity import DataIngestionArtifact 

## Configuration of Data Ingestion Config 
from energygeneration.entity.config_entity import DataIngestionConfig
from typing import List
import pandas as pd
import pymongo
import os 
import sys
import numpy as np

from dotenv import load_dotenv
load_dotenv()

MONGO_DB_URL = os.getenv("PYMONGO_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise EnergyGenerationException(e,sys)

    def fetch_data_from_mongodb(self):
        """Reading the data from mongodb and exporting as a dataframe
        """
        try:
            logging.info("Fetching the data from Mongodb.")
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))
            cols = ['_id', 'respondent', 'respondent-name', 'fueltype', 'type-name', 'value-units']

            # Check if all columns in 'cols' are in the DataFrame columns
            if set(cols).issubset(df.columns):
                df.drop(columns=cols, inplace=True)  

            df.replace({"na":np.nan},inplace = True)
            # Ensure the "period" column is parsed as dates and set as the index
            if "period" in df.columns:
                df['period'] = pd.to_datetime(df['period'], errors="coerce") 
                 
                df.sort_values(by='period', inplace=True)
            logging.info("completed fetching the data from mongodb.")
            return df
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    

    def export_data_to_dataframe(self, dataframe: pd.DataFrame):
        try:
            logging.info("Converting the fetched data into a dataframe.")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            # Create the directory if it doesn't exist
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            dataframe.to_csv(feature_store_file_path,index=False,header=True)
            logging.info("The Energy data has been saved in feature store file path.")
            return dataframe
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    
    def train_val_test_split(self, dataframe: pd.DataFrame):
        """
        Splits the dataset into train, validation, and test sets and saves them as CSV files.
        """
        try:
            logging.info("Starting data split into train, validation, and test sets.")
            
            # Calculate split indices
            total_samples = len(dataframe)
            train_size = int(total_samples * self.data_ingestion_config.train_val_test_split_ratio)
            remaining_size = total_samples - train_size  # Remaining 30%
            val_size = int(remaining_size * self.data_ingestion_config.validation_split_ratio)  # Validation is 40% of the remaining 30%
 
            # Perform the splits
            train_set = dataframe[:train_size]
            val_set = dataframe[train_size:train_size + val_size]
            test_set = dataframe[train_size + val_size:]
            # Reset indices for train, validation, and test sets
            train_set = train_set.reset_index(drop=True)
            val_set = val_set.reset_index(drop=True)
            test_set = test_set.reset_index(drop=True)
            logging.info("Completed splitting the data into train, validation, and test sets.")
 
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)        
            # Save the splits as CSV files
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False,header=True)
            val_set.to_csv(self.data_ingestion_config.validation_file_path, index=False,header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index=False,header=True)

            logging.info("Train, validation, and test datasets saved as CSV files.")

            print("Data split sizes:")
            print(f"Train set: {len(train_set)}, Validation set: {len(val_set)}, Test set: {len(test_set)}")

            return train_set, val_set, test_set

        except Exception as e:
            raise EnergyGenerationException(e, sys)
        
    
    def ingest_data(self)-> DataIngestionArtifact:
        try:
            dataframe = self.fetch_data_from_mongodb()
            dataframe = self.export_data_to_dataframe(dataframe)
            self.train_val_test_split(dataframe)
            dataingestionartifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path,
                                                          test_file_path= self.data_ingestion_config.testing_file_path,
                                                          val_file_path=self.data_ingestion_config.validation_file_path)
            return dataingestionartifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)