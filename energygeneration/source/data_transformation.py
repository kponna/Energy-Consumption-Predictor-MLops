import sys 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from energygeneration.logging.logger import logging
from energygeneration.constant import SCHEMA_FILE_PATH 
from energygeneration.constant import TARGET_COLUMN, TIME_STEPS
from energygeneration.entity.config_entity import DataTransformationConfig
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from energygeneration.utils.main_utils.utils import save_numpy_array_data, add_cyclic_features, df_to_X_y, scale_and_save_target

class DataTransformation:
    def __init__(self,data_validation_artifact:DataValidationArtifact,
                 data_transformation_config:DataTransformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config
        except Exception as e:
            raise EnergyGenerationException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise EnergyGenerationException(e,sys)

 

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Entering  initiate_data_transformation method of DataTransformation class.")
        try:
            logging.info("Starting Data Transformation.")

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

            # Splitting the features and target
            input_feature_train_df = train_df.drop(columns = [TARGET_COLUMN],axis = 1)
            target_feature_train_df = pd.DataFrame(train_df[TARGET_COLUMN],columns=[TARGET_COLUMN])

            input_feature_test_df = test_df.drop(columns = [TARGET_COLUMN],axis = 1)
            target_feature_test_df = pd.DataFrame( test_df[TARGET_COLUMN],columns=[TARGET_COLUMN])

            input_feature_val_df = val_df.drop(columns = [TARGET_COLUMN],axis = 1)
            target_feature_val_df = pd.DataFrame(val_df[TARGET_COLUMN],columns=[TARGET_COLUMN])

            for df_name, df in {
                "Input Train": input_feature_train_df, "Input Test": input_feature_test_df, "Input Val": input_feature_val_df,
                "Target Train": target_feature_train_df,"Target Test":target_feature_test_df,"Target Val": target_feature_val_df
            }.items():
                logging.info(f"{df_name} Head:\n{df.head()}\n")
                logging.info(f"{df_name} Shape: {df.shape}\n")
            
            # Initialize MinMaxScaler
            scaler = MinMaxScaler()
            scaler_path = self.data_transformation_config.transformed_object_file_path
            # Scale and save target features
            y_train_scaled = scale_and_save_target(
                scaler=scaler, 
                target_df=target_feature_train_df, 
                scaler_file_path=scaler_path
            )
            logging.info(f"y_train scaled: {y_train_scaled.shape}")
            y_val_scaled = scaler.transform(target_feature_val_df)
            logging.info(f"y_val scaled: {y_val_scaled.shape}")
            y_test_scaled = scaler.transform(target_feature_test_df)
            logging.info(f"y_test scaled: {y_test_scaled.shape}") 

            # Create DataFrames from scaled arrays 
            y_train_scaled_df = pd.DataFrame(y_train_scaled, columns=[TARGET_COLUMN]) 
            combined_train_df = pd.concat([input_feature_train_df, y_train_scaled_df], axis=1)
            logging.info(f"scaled train df :{combined_train_df.head()}")

            # Create DataFrames from scaled arrays 
            y_test_scaled_df = pd.DataFrame(y_test_scaled, columns=[TARGET_COLUMN]) 
            combined_test_df = pd.concat([input_feature_test_df, y_test_scaled_df], axis=1)
            logging.info(f"scaled test df :{combined_test_df.head()}")

            # Create DataFrames from scaled arrays 
            y_val_scaled_df = pd.DataFrame(y_val_scaled, columns=[TARGET_COLUMN]) 
            combined_val_df = pd.concat([input_feature_val_df, y_val_scaled_df], axis=1)
            logging.info(f"scaled val df :{combined_val_df.head()}")


            # Converting the dataframe to numpy array
            logging.info("Applying df_to_X_y function on datasets.")
            input_feature_train_np,target_feature_train_np = df_to_X_y(combined_train_df,time_steps=TIME_STEPS)
            input_feature_test_np, target_feature_test_np = df_to_X_y(combined_test_df,time_steps=TIME_STEPS)
            input_feature_val_np, target_feature_val_np = df_to_X_y(combined_val_df,time_steps=TIME_STEPS)
            logging.info(f"input_feature_train_np.shape:{input_feature_train_np.shape}, input_feature_test_np.shape: {input_feature_test_np.shape},input_feature_val_np.shape: {input_feature_val_np.shape}")
            logging.info(f"target_feature_train_np.shape:{target_feature_train_np.shape},target_feature_test_np.shape: {target_feature_test_np.shape},target_feature_val_np.shape: {target_feature_val_np.shape}")
            
            # Saving the numpy arrays
            save_numpy_array_data(self.data_transformation_config.transformed_X_train_file_path,input_feature_train_np)
            save_numpy_array_data(self.data_transformation_config.transformed_X_test_file_path,input_feature_test_np)
            save_numpy_array_data(self.data_transformation_config.transformed_X_val_file_path,input_feature_val_np)
            save_numpy_array_data(self.data_transformation_config.transformed_y_train_file_path,target_feature_train_np)
            save_numpy_array_data(self.data_transformation_config.transformed_y_test_file_path,target_feature_test_np)
            save_numpy_array_data(self.data_transformation_config.transformed_y_val_file_path,target_feature_val_np)

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