import os
import sys 
import pandas as pd  
from scipy.stats import ks_2samp

from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.config_entity import DataValidationConfig
from energygeneration.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from energygeneration.constant import SCHEMA_FILE_PATH
from energygeneration.utils.main_utils.utils import read_yaml_file,write_yaml_file
class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH) 
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EnergyGenerationException(e,sys)
        
    def validate_no_of_cols(self,dataframe:pd.DataFrame):
        try:
            no_of_cols = len(self._schema_config["columns"])# len(self._schema_config)
            logging.info(f"Required no. of columns:{no_of_cols}")
            logging.info(f"Dataframe  has columns: {len(dataframe.columns)}")
            if len(dataframe.columns) == no_of_cols:
                return True
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    
    def is_column_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates the existence of numerical and datetime columns.
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

    # def is_num_cols_exist(self,dataframe:pd.DataFrame):
    #     try:
    #         num_cols = self._schema_config['numerical_columns']
    #         dataframe_columns = dataframe.columns
    #         num_col_exist = True
    #         missing_num_cols = []
    #         for num_col in num_cols:
    #             if num_col not in dataframe_columns:
    #                 num_col_exist = False
    #                 missing_num_cols.append(num_col)
    #         if missing_num_cols:
    #             logging.info(f"Missing numerical columns: {missing_num_cols}")
    #             return False
    #         return num_col_exist
    #     except Exception as e:
    #         raise EnergyGenerationException(e,sys)

    def detect_data_drift(self, base_df, current_df, threshold= 0.05)->bool:
        try:
            # status = True
            drift_report = {}
            for col in base_df.columns:
                d1 = base_df[col]
                d2 = current_df[col]
                is_same_distance = ks_2samp(d1,d2)
                if threshold<= is_same_distance.pvalue:
                    is_found = False
                else:
                    is_found = True
                    # status = False
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
            
            # # Validating numerical columns
            # if not self.is_num_cols_exist(train_dataframe):
            #     error_message += "Train dataframe is missing required numerical columns.\n"
            # if not self.is_num_cols_exist(test_dataframe):
            #     error_message += "Test dataframe is missing required numerical columns.\n"
            # if not self.is_num_cols_exist(val_dataframe):
            #     error_message += "Validation dataframe is missing required numerical columns.\n"

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
         
