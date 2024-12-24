from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.artifact_entity import DataIngestionArtifact
from energygeneration.entity.config_entity import TrainingPipelineConfig
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
                df.drop(columns=cols, inplace=True)  # Drop the columns

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
  
    def df_to_X_y(self, dataframe: pd.DataFrame):
        """
        Converts a DataFrame into X and y datasets for time series modeling.
        """
        try:
            window_size = self.data_ingestion_config.window_size
            df_as_np = dataframe.to_numpy()
            X = []
            y = []

            for i in range(len(df_as_np) - window_size):
                # Use all columns for X
                row = df_as_np[i:i + window_size]
                X.append(row)
                # Only use the target column ('value') for y
                label = df_as_np[i + window_size][0]
                y.append(label)

            return np.array(X), np.array(y)
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    
    def train_val_test_split(self, dataframe: pd.DataFrame):
        """
        Splits the dataset into train, validation, and test sets and saves them as CSV files.
        """
        try:
            logging.info("Starting data split into train, validation, and test sets.")
            
            # Calculate split indices
            total_samples = len(dataframe)
            train_size = int(total_samples * self.data_ingestion_config.train_val_test_split_ratio)
            val_size = int((total_samples - train_size) * self.data_ingestion_config.validation_split_ratio)

            # Perform the splits
            train_set = dataframe[:train_size]
            val_set = dataframe[train_size:train_size + val_size]
            test_set = dataframe[train_size + val_size:]

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
        

if __name__=="__main__":
    # Initialize data_ingestion_config and DataIngestion class
    training_pipeline_config = TrainingPipelineConfig()
    data_ingestion_config = DataIngestionConfig(training_pipeline_config)
    data_ingestion = DataIngestion(data_ingestion_config)
    dataframe = pd.DataFrame({  # Example DataFrame
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'target': np.random.randint(0, 2, 100)
    })

    # Call the method
    train_set, val_set, test_set = data_ingestion.train_val_test_split(dataframe)

    # Verify the splits
    print(f"Train size: {len(train_set[0])}, Validation size: {len(val_set[0])}, Test size: {len(test_set[0])}")


    # def train_val_test_split(self, dataframe: pd.DataFrame):
    #     """
    #     Splits the dataset into train, validation, and test sets after converting it to X and y using df_to_X_y.
    #     """
    #     try:
    #         logging.info("Converting DataFrame to X and y datasets.")
    #         # Convert DataFrame to X, y
    #         X, y = self.df_to_X_y(dataframe)

    #         # Calculate split indices
    #         total_samples = len(X)
    #         train_size = int(total_samples * self.data_ingestion_config.train_val_test_split_ratio)
    #         val_size = int((total_samples - train_size) * self.data_ingestion_config.validation_split_ratio)

    #         logging.info("Starting data split into train, validation, and test sets.")

    #         # Perform the splits
    #         X_train, y_train = X[:train_size], y[:train_size]
    #         X_val, y_val = X[train_size:train_size + val_size], y[train_size:train_size + val_size]
    #         X_test, y_test = X[train_size + val_size:], y[train_size + val_size:]

    #         logging.info("Completed splitting the data.")

    #         # Save the splits
    #         dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
    #         os.makedirs(dir_path, exist_ok=True)

    #         # Save train, val, and test sets as separate NumPy arrays
    #         np.save(self.data_ingestion_config.training_file_path + "_X.npy", X_train)
    #         np.save(self.data_ingestion_config.training_file_path + "_y.npy", y_train)
    #         np.save(self.data_ingestion_config.validation_file_path + "_X.npy", X_val)
    #         np.save(self.data_ingestion_config.validation_file_path + "_y.npy", y_val)
    #         np.save(self.data_ingestion_config.testing_file_path + "_X.npy", X_test)
    #         np.save(self.data_ingestion_config.testing_file_path + "_y.npy", y_test)

    #         logging.info("Train, validation, and test datasets saved as separate .npy files.")

    #         print("Data split sizes:")
    #         print(f"Train set: {len(X_train)}, Validation set: {len(X_val)}, Test set: {len(X_test)}")

    #         return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    #     except Exception as e:
    #         raise EnergyGenerationException(e, sys)



    # def train_val_test_split(self, dataframe: pd.DataFrame):
    #     try:
    #         # Convert the DataFrame to X and y datasets
    #         FEATURES, TARGET = self.df_to_X_y(dataframe)
    #         print("X shape:", FEATURES.shape)  # Should be (n_samples, window_size, n_features)
    #         print("y shape:", TARGET.shape)  # Should be (n_samples,)

    #         # Split the data into training, validation, and test sets
    #         training_limit = int(len(FEATURES) * self.data_ingestion_config.train_val_test_split_ratio)
    #         print(f'training_limit: {training_limit}')
    #         rest = len(FEATURES) - training_limit
    #         print(f'remaining: {rest}')

    #         validation_limit = int(rest * self.data_ingestion_config.validation_split_ratio)
    #         print(f'validation_limit: {validation_limit}')

    #         logging.info("Starting the data splitting into train, val, test sets")

    #         # Generating training, validation, and test sets
    #         X_train, y_train = FEATURES[:training_limit], TARGET[:training_limit]
    #         X_val, y_val = FEATURES[training_limit:(training_limit + validation_limit)], TARGET[training_limit:(training_limit + validation_limit)]
    #         X_test, y_test = FEATURES[(training_limit + validation_limit):], TARGET[(training_limit + validation_limit):]
    #         logging.info("Completed splitting the data.")
    #         dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
    #         os.makedirs(dir_path,exist_ok=True)
    #         logging.info("Exporting train , validation, test file paths.")

    #         X_train.to_csv(self.data_ingestion_config.training_file_path,index = True,  header = True)
    #         y_train.to_csv(self.data_ingestion_config.training_file_path,index = True,  header = True)
    #         X_val.to_csv(self.data_ingestion_config.validation_file_path,index = True,  header = True)
    #         y_val.to_csv(self.data_ingestion_config.validation_file_path,index = True,  header = True)
    #         X_test.to_csv(self.data_ingestion_config.testing_file_path,index = True,  header = True)
    #         y_test.to_csv(self.data_ingestion_config.testing_file_path,index = True,  header = True)
    #         logging.info("Exported the train, validation, test files.")
            
    #         print("Training, Validation, and Test Shapes:")
    #         print(X_train.shape, y_train.shape, X_val.shape, y_val.shape, X_test.shape, y_test.shape)

    #         return X_train, y_train, X_val, y_val, X_test, y_test
    #     except Exception as e:
    #         raise EnergyGenerationException(e, sys) 
    
    # def train_val_test_split(self, dataframe: pd.DataFrame):
    #     """
    #     Splits the dataset into train, validation, and test sets after converting it to X and y using df_to_X_y.
    #     """
    #     try:
    #         logging.info("Converting DataFrame to X and y datasets.")
    #         # Convert DataFrame to X, y
    #         X, y = self.df_to_X_y(dataframe)

    #         # Calculate split indices
    #         total_samples = len(X)
    #         train_size = int(total_samples * self.data_ingestion_config.train_val_test_split_ratio)
    #         val_size = int((total_samples - train_size) * self.data_ingestion_config.validation_split_ratio)

    #         logging.info("Starting data split into train, validation, and test sets.")

    #         # Perform the splits
    #         train_set = (X[:train_size], y[:train_size])
    #         val_set = (X[train_size:train_size + val_size], y[train_size:train_size + val_size])
    #         test_set = (X[train_size + val_size:], y[train_size + val_size:])

    #         logging.info("Completed splitting the data.")

    #         # Save the splits
    #         dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
    #         os.makedirs(dir_path, exist_ok=True)

    #         # Save train, val, and test sets as NumPy arrays 
    #         np.save(self.data_ingestion_config.training_file_path, train_set)
    #         np.save(self.data_ingestion_config.validation_file_path, val_set)
    #         np.save(self.data_ingestion_config.testing_file_path, test_set)

    #         logging.info("Train, validation, and test datasets saved as .npy files.")

    #         print("Data split sizes:")
    #         print(f"Train set: {len(train_set[0])}, Validation set: {len(val_set[0])}, Test set: {len(test_set[0])}")

    #         return train_set, val_set, test_set
    #     except Exception as e:
    #         raise EnergyGenerationException(e, sys)