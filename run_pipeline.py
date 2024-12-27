from energygeneration.source.data_ingestion import DataIngestion
from energygeneration.source.data_validation import DataValidation
from energygeneration.source.data_transformation import DataTransformation
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig
from energygeneration.entity.config_entity import TrainingPipelineConfig

import sys

if __name__ == "__main__":
    try:
        training_pipeline_config = TrainingPipelineConfig()

        # Data Ingestion
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiating Data Ingestion. ") 
        data_ingestion_artifact = data_ingestion.ingest_data()
        logging.info(f"Data ingestion completed.")
        print(data_ingestion_artifact)

        # Data Validation
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact,data_validation_config)
        logging.info("Initiating Data Validation.")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data Validation completed.")
        print(data_validation_artifact)

        # Data Transformation
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_validation_artifact,data_transformation_config)
        logging.info("Initiating Data Transformation.")
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info(" Data Transformation Completed.")
        print(data_transformation_artifact)

    except Exception as e:
        raise EnergyGenerationException(e,sys)