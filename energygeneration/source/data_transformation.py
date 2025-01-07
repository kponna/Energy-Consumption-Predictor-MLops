import sys 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from energygeneration.logging.logger import logging
from energygeneration.constant import SCHEMA_FILE_PATH 
from energygeneration.constant import TIME_STEPS
from energygeneration.entity.config_entity import DataTransformationConfig
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from energygeneration.utils.main_utils.utils import save_numpy_array_data, add_cyclic_features, df_to_X_y, save_object

class DataTransformation:
    """
    Class for handling data transformation for energy generation models.
    """
    def __init__(self,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTransformationConfig):
        """
        Initializes the DataTransformation class with the given data validation and configuration artifacts.
        
        Args:
            data_validation_artifact (DataValidationArtifact): The artifact containing validated data paths.
            data_transformation_config (DataTransformationConfig): Configuration object for data transformation.
        """
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
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
 
    def initiate_data_transformation(self)->DataTransformationArtifact:
        """
        Initiates the data transformation process by reading, transforming, and saving the data.

        Returns:
            DataTransformationArtifact: The artifact containing paths to the transformed data. 
        """
        logging.info("Entering initiate_data_transformation method of DataTransformation class.")
        try: 
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            val_df = DataTransformation.read_data(self.data_validation_artifact.valid_val_file_path)
            logging.info(f"Train df :{train_df.head()}")
            logging.info(f"Test df: {test_df.head()}")
            logging.info(f"Val df: {val_df.head()}")
            logging.info(f"train_df.shape:{train_df.shape}, test_df.shape: {test_df.shape},val_df.shape: {val_df.shape}") 
            logging.info(f"train cols:{train_df.columns},test cols: {test_df.columns},val cols: {val_df.columns}")

            # Applying cyclic feature transformation
            logging.info("Applying cyclic feature transformation on datasets.")
            train_df = add_cyclic_features(train_df, SCHEMA_FILE_PATH)
            test_df = add_cyclic_features(test_df, SCHEMA_FILE_PATH)
            val_df = add_cyclic_features(val_df, SCHEMA_FILE_PATH)
            logging.info(f"Train df :{train_df.head()}")
            logging.info(f"Test df: {test_df.head()}")
            logging.info(f"Val df: {val_df.head()}")
            logging.info(f"train_df.shape:{train_df.shape}, test_df.shape: {test_df.shape},val_df.shape: {val_df.shape}") 
            logging.info(f"train cols:{train_df.columns},test cols: {test_df.columns},val cols: {val_df.columns}")
            logging.info("splitting the dataframes into input and target dataframes.")

            X_train, y_train = df_to_X_y(train_df, time_steps=TIME_STEPS)
            X_test, y_test = df_to_X_y(test_df, time_steps=TIME_STEPS)
            X_val, y_val = df_to_X_y(val_df, time_steps=TIME_STEPS)
            logging.info(f"X_train.shape:{X_train.shape}, X_test.shape:{X_test.shape}, X_val.shape:{X_val.shape}")
            logging.info(f"y_train.shape:{y_train.shape}, y_test.shape:{y_test.shape}, y_val.shape:{y_val.shape}") 
 
            scaler = MinMaxScaler()
            y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)) 
            y_val_scaled = scaler.transform(y_val.reshape(-1, 1))
            y_test_scaled = scaler.transform(y_test.reshape(-1, 1)) 
            logging.info("Printing First 5 elements:")
            logging.info(f"y_train_scaled: {y_train_scaled[:5]}, y_val_scaled: {y_val_scaled[:5]}, y_test_scaled: {y_test_scaled[:5]}") 
            logging.info("Printing shapes:")
            logging.info(f"y_train_scaled: {y_train_scaled.shape}, y_val_scaled: {y_val_scaled.shape}, y_test_scaled: {y_test_scaled.shape}")
            
            save_object(self.data_transformation_config.transformed_object_file_path,scaler)
            save_object("final_model/scaler.pkl",scaler)

            # Save the numpy arrays for X and scaled y
            save_numpy_array_data(self.data_transformation_config.transformed_X_train_file_path, X_train)
            save_numpy_array_data(self.data_transformation_config.transformed_X_test_file_path, X_test)
            save_numpy_array_data(self.data_transformation_config.transformed_X_val_file_path, X_val)
            save_numpy_array_data(self.data_transformation_config.transformed_y_train_file_path, y_train_scaled)
            save_numpy_array_data(self.data_transformation_config.transformed_y_test_file_path, y_test_scaled)
            save_numpy_array_data(self.data_transformation_config.transformed_y_val_file_path, y_val_scaled)
             
            logging.info("saved numpy arrays in the path.")

            data_transformation_artifact = DataTransformationArtifact( 
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_X_train_file_path=self.data_transformation_config.transformed_X_train_file_path,
                transformed_X_test_file_path=self.data_transformation_config.transformed_X_test_file_path,
                transformed_X_val_file_path=self.data_transformation_config.transformed_X_val_file_path,
                transformed_y_train_file_path=self.data_transformation_config.transformed_y_train_file_path,
                transformed_y_test_file_path=self.data_transformation_config.transformed_y_test_file_path,
                transformed_y_val_file_path=self.data_transformation_config.transformed_y_val_file_path, 
                            )
            
            return data_transformation_artifact
        except Exception as e:
            raise EnergyGenerationException(e,sys)