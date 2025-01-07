import os
import sys 
import pandas as pd  
from scipy.stats import ks_2samp

from energygeneration.logging.logger import logging
from energygeneration.exception_handling.exception import EnergyGenerationException

from energygeneration.constant import SCHEMA_FILE_PATH
from energygeneration.utils.main_utils.utils import read_yaml_file,write_yaml_file

from energygeneration.entity.config_entity import DataValidationConfig
from energygeneration.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
 
class DataValidation:
    """
    Class for handling data validation and data drift.
    """
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact, data_validation_config: DataValidationConfig):
        """
        Initializes the DataValidation class.

        Args:
            data_ingestion_artifact (DataIngestionArtifact): The artifact containing paths to the ingested data.
            data_validation_config (DataValidationConfig): Configuration object for data validation. 
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH) 
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        """
        Reads data from a CSV file.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The DataFrame containing the data from the CSV file. 
        """
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    def validate_no_of_cols(self,dataframe:pd.DataFrame):
        """
        Validates if the DataFrame contains the required number of columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if the number of columns matches the expected count, False otherwise. 
        """
        try:
            no_of_cols = len(self._schema_config["columns"]) 
            logging.info(f"Required no. of columns:{no_of_cols}")
            logging.info(f"Dataframe  has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == no_of_cols:
                return True
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    
    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the existence of required numerical and datetime columns.

        Args:
            dataframe (pd.DataFrame): The DataFrame to validate.

        Returns:
            bool: True if all required numerical and datetime columns exist, False otherwise. 
        """
        try:
            dataframe_columns = dataframe.columns
            missing_numerical_columns = []
            missing_datetime_columns = []

            # Validate numerical columns
            for column in self._schema_config["numerical_columns"]:
                if column not in dataframe_columns:
                    missing_numerical_columns.append(column)

            if missing_numerical_columns:
                logging.info(f"Missing numerical columns: {missing_numerical_columns}")

            # Validate datetime columns
            for column in self._schema_config["datetime_columns"]:
                if column not in dataframe_columns:
                    missing_datetime_columns.append(column)

            if missing_datetime_columns:
                logging.info(f"Missing datetime columns: {missing_datetime_columns}")

            # Return False if any columns are missing
            return len(missing_numerical_columns) == 0 and len(missing_datetime_columns) == 0

        except Exception as e:
            raise EnergyGenerationException(e, sys) 

    def detect_data_drift(self, base_df, current_df, threshold= 0.05) ->bool:
        """
        Detects data drift between the base and current DataFrames.

        Args:
            base_df (pd.DataFrame): The base DataFrame to compare against.
            current_df (pd.DataFrame): The current DataFrame to check for drift.
            threshold (float): The p-value threshold to consider drift. Default is 0.05.

        Returns:
            bool: True if no drift is detected (i.e., p-value > threshold), False otherwise. 
        """
        try: 
            drift_report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                is_same_distance = ks_2samp(d1,d2)
                if threshold<= is_same_distance.pvalue:
                    is_found = False
                else:
                    is_found = True 
                drift_report.update({col:{
                    "p_value":float(is_same_distance.pvalue),
                    "drift_status":is_found
                }})
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path,exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=drift_report)
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    def initiate_data_validation(self)-> DataValidationArtifact:
        """
        Initiates the data validation process by reading, validating, and saving data.

        Returns:
            DataValidationArtifact: The artifact containing paths to the validated data. 
        """
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            val_file_path = self.data_ingestion_artifact.val_file_path

            # Read the data from the paths
            train_dataframe = DataValidation.read_data(train_file_path)
            test_dataframe = DataValidation.read_data(test_file_path)
            val_dataframe = DataValidation.read_data(val_file_path)
 
            # Validating columns
            error_message = ""
            if not self.validate_no_of_cols(train_dataframe):
                error_message += "Train dataframe does not have required columns.\n"
            if not self.validate_no_of_cols(test_dataframe):
                error_message += "Test dataframe does not have required columns.\n"
            if not self.validate_no_of_cols(val_dataframe):
                error_message += "Validation dataframe does not have required columns.\n"
             
            # Validate existence of numerical and datetime columns
            if not self.is_column_exist(train_dataframe):
                error_message += "Train dataframe is missing required numerical or datetime columns.\n"
            if not self.is_column_exist(test_dataframe):
                error_message += "Test dataframe is missing required numerical or datetime columns.\n"
            if not self.is_column_exist(val_dataframe):
                error_message += "Validation dataframe is missing required numerical or datetime columns.\n"

            if error_message:
                raise ValueError(error_message) 

            # check datadrift
            test_drift_status = self.detect_data_drift(base_df=train_dataframe,current_df=test_dataframe)
            print(f"Test drift status:{test_drift_status}")
            val_drift_status = self.detect_data_drift(base_df=train_dataframe, current_df=val_dataframe)
            print(f"Validation drift status:{val_drift_status}")
            drift_status = test_drift_status and val_drift_status
            dir_path = os.path.dirname(self.data_validation_config.valid_train_file_path)
            os.makedirs(dir_path,exist_ok=True)
             
            train_dataframe.to_csv(self.data_validation_config.valid_train_file_path,index=False, header=True)
            test_dataframe.to_csv(self.data_validation_config.valid_test_file_path,index=False, header=True)
            val_dataframe.to_csv(self.data_validation_config.valid_val_file_path, index=False, header=True)
              
            data_validation_artifact = DataValidationArtifact(
                validation_status=drift_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                valid_val_file_path=self.data_validation_config.valid_val_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                invalid_val_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            ) 
            return data_validation_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)
         
