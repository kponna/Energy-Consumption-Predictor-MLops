import os
import sys

from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging

from energygeneration.entity.config_entity import (
    TrainingPipelineConfig,
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from energygeneration.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from energygeneration.source.data_ingestion import DataIngestion
from energygeneration.source.data_validation import DataValidation
from energygeneration.source.data_transformation import DataTransformation
from energygeneration.source.model_trainer import ModelTrainer
from energygeneration.constant import TRAINING_BUCKET_NAME

from energygeneration.cloud.s3_syncer import S3Sync

class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config=TrainingPipelineConfig()
        self.s3_sync = S3Sync()
    def start_data_ingestion(self):
        try:
            self.data_ingestion_config=DataIngestionConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data Ingesiton")
            data_ingestion=DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.ingest_data()
            logging.info(f"Data Ingeston completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)

    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact):
        try:
            data_validation_config=DataValidationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data validation")
            data_validation=DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config=data_validation_config )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    def start_data_transformation(self,data_validation_artifact:DataValidationArtifact):
        try:
            data_transformation_config=DataTransformationConfig(training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(data_validation_artifact=data_validation_artifact,data_transformation_config=data_transformation_config)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}") 
            return data_transformation_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    def start_model_trainer(self,data_transformation_artifact: DataTransformationArtifact):
        try:
            model_trainer_config = ModelTrainerConfig(training_pipeline_config=self.training_pipeline_config)
            model_trainer = ModelTrainer(model_trainer_config= model_trainer_config,data_transformation_artifact=data_transformation_artifact)
            logging.info("Initiating Model training.")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    ## local artifact is pushed to s3 bucket
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url= f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    ## local final model is pushed to s3 bucket
    def sync_save_model_dir_to_s3(self):
        try:
            aws_bucket_url= f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir,aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    def run_pipeline(self):
        try:
            data_ingestion_artifact=self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)
            data_transformation_artifact=self.start_data_transformation(data_validation_artifact=data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact=data_transformation_artifact)
           
            self.sync_artifact_dir_to_s3()
            self.sync_save_model_dir_to_s3()
            
            return model_trainer_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)