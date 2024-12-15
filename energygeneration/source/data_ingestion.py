import os
import sys
import logging
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv

from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.config_entity import TrainingPipelineConfig
from energygeneration.entity.config_entity import DataIngestionConfig
from energygeneration.entity.artifact_entity import DataIngestionArtifact

# Load the .env file
load_dotenv()

# Get MongoDB URI from environment variables
MONGO_URI = os.getenv("PYMONGO_URI")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def fetch_data_from_mongodb(self, collection_name: str, database_name: str):  
        """Fetches data from MongoDB and converts it into a pandas DataFrame"""
        # Connect to the MongoDB client
        client = MongoClient(MONGO_URI)
        db = client[database_name]  
        collection = db[collection_name]  

        try:
            logging.info(f"Fetching data from MongoDB collection: {collection_name}...")
            data = list(collection.find())  # Convert cursor to a list of dictionaries

            if not data:
                logging.info("No data found in the MongoDB collection.")
                return pd.DataFrame()  # Return empty DataFrame if no data is found

            # Convert the list of dictionaries into a pandas DataFrame
            df = pd.DataFrame(data)
 
            if '_id' in df.columns:
                df = df.drop(columns=['_id'])

            logging.info("Data successfully fetched and converted to a DataFrame!")
            return df

        except Exception as e:
            logging.error(f"An error occurred while fetching data: {e}")
            raise EnergyGenerationException(e, sys)

    def ingest_data(self):
        """Ingests data from MongoDB and stores it in the feature store directory"""
        try:
            # Fetch data from MongoDB
            df = self.fetch_data_from_mongodb(
                collection_name=self.data_ingestion_config.collection_name,
                database_name=self.data_ingestion_config.database_name
            )

            # If data is empty, raise an exception
            if df.empty:
                raise EnergyGenerationException("No data found in the MongoDB collection.")

            # Ensure the directory exists before saving the file
            os.makedirs(os.path.dirname(self.data_ingestion_config.feature_store_file_path), exist_ok=True)

            # Save data to feature store path
            df.to_csv(self.data_ingestion_config.feature_store_file_path, index=False)
            logging.info(f"Data successfully saved to {self.data_ingestion_config.feature_store_file_path}")

            # Create DataIngestionArtifact object
            data_ingestion_artifact = DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path
            ) 
            return data_ingestion_artifact

        except Exception as e:
            logging.error(f"An error occurred during data ingestion: {e}")
            raise EnergyGenerationException(e, sys)


if __name__ == "__main__":
    try:
        # Initialize DataIngestionConfig and DataIngestion
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)

        # Perform data ingestion
        data_ingestion_artifact = data_ingestion.ingest_data()
        logging.info(f"Data ingestion completed. Feature store file path: {data_ingestion_artifact.feature_store_file_path}")

    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")
        raise EnergyGenerationException(e,sys)
