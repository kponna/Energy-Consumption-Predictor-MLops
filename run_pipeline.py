# from energygeneration.source.data_ingestion0 import DataIngestion
from energygeneration.source.data_ingestion import DataIngestion
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.config_entity import DataIngestionConfig
from energygeneration.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating Data Ingestion. ")
        # Perform data ingestion
        data_ingestion_artifact = data_ingestion.ingest_data()
        logging.info(f"Data ingestion completed.")
         
        print(data_ingestion_artifact)
    except Exception as e:
        raise EnergyGenerationException(e,sys)