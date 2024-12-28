from energygeneration.source.data_ingestion import DataIngestion
from energygeneration.source.data_validation import DataValidation
from energygeneration.source.model_trainer import ModelTrainer
from energygeneration.source.data_transformation import DataTransformation
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
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
        print(data_transformation_artifact)
        logging.info(" Data Transformation Completed.")

        # Model Training
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config= model_trainer_config,data_transformation_artifact=data_transformation_artifact)
        logging.info("Initiating Model training.")
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        logging.info("Model training completed.")


    except Exception as e:
        raise EnergyGenerationException(e,sys)