import yaml
import os,sys
import pickle 
import numpy as np
import pandas as pd 
from datetime import timedelta
from energygeneration.constant import TIME_STEPS
from energygeneration.logging.logger import logging
from energygeneration.exception_handling.exception import EnergyGenerationException
 
def read_yaml_file(file_path:str) -> dict:
    """
    Reads a YAML file and returns its contents as a dictionary. 
    Args:
        file_path (str): Path to the YAML file. 
    Returns:
        dict: Parsed contents of the YAML file. 
    """
    try:
        with open(file_path,"rb") as yaml_file: 
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e 

def write_yaml_file(file_path:str, content:object, replace:bool = False) -> None:
    """
    Writes content to a YAML file. 
    Args:
        file_path (str): Path to the YAML file.
        content (object): Content to write to the file.
        replace (bool, optional): Whether to replace the file if it already exists. Defaults to False. 
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content,file) 
    except Exception as e:
        raise EnergyGenerationException(e,sys)
    
def save_numpy_array_data(file_path:str, array:np.array)-> None:
    """
    Saves a NumPy array to a file. 
    Args:
        file_path (str): Path where the array will be saved.
        array (np.array): NumPy array to save. 
    Returns:
        None
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,"wb") as file_obj:
            np.save(file_obj,array)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e

def load_numpy_array_data(file_path:str)-> np.array:
    """
    Loads a NumPy array from a file.
    Args:
        file_path (str): Path to the file containing the NumPy array.
    Returns:
        np.array: Loaded NumPy array.
    """
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e
    
def save_object(file_path:str,obj:object)-> None:
    """
    Saves an object to a file using pickle.
    Args:
        file_path (str): Path where the object will be saved.
        obj (object): Object to save.
    Returns:
        None
    """
    try:
        logging.info("Entered the save_object method of Main utils class")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open (file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Completed saving the scaling object.")
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e

def load_object(file_path:str,)-> object:
    """
    Loads an object from a file using pickle.
    Args:
        file_path (str): Path to the file containing the object.
    Returns:
        object: Loaded object.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e
    
def add_cyclic_features(data: pd.DataFrame, schema_file: str) -> pd.DataFrame:
    """
    Adds cyclic features (sin and cos transformations for time-based data) to a DataFrame. 
    Args:
        data (pd.DataFrame): The input DataFrame containing time-based data.
        schema_file (str): Path to the schema YAML file containing configurations for cyclic features. 
    Returns:
        pd.DataFrame: The DataFrame with added cyclic features.
    """
    try:
        schema = read_yaml_file(schema_file)
        df = data.copy() 
        
        # Load column names and constants from schema
        input_column = schema["cyclic_features"]["input_column"]
        timestamp_column = schema["cyclic_features"]["timestamp_column"]
        datetime_column = schema['cyclic_features']['datetime_column']
        day_features = schema["cyclic_features"]["day_features"]
        year_features = schema["cyclic_features"]["year_features"]
        hour_features = schema["cyclic_features"]["hour_features"]
        minute_features = schema["cyclic_features"]["minute_features"]
        month_features = schema["cyclic_features"]["month_features"]
        drop_columns = schema["drop_columns"]

        # Convert the 'period' column to datetime if not already
        df[input_column] = pd.to_datetime(df[input_column])

        # Convert datetime to timestamp for calculations
        df[timestamp_column] = df[input_column].map(pd.Timestamp.timestamp)
        # Constants for cyclic features
        day = 60 * 60 * 24
        year = timedelta(days=365.2425).total_seconds()
        hour = 60 * 60
        minute = 60 
        # Cyclic features for day and year
        df[day_features[0]] = np.sin(2 * np.pi * df[timestamp_column] / day)
        df[day_features[1]] = np.cos(2 * np.pi * df[timestamp_column] / day)
        df[year_features[0]] = np.sin(2 * np.pi * df[timestamp_column] / year)
        df[year_features[1]] = np.cos(2 * np.pi * df[timestamp_column] / year)

        # Extract hours, minutes, and months from the datetime column 
        df["Hour"] = (df[timestamp_column] // hour) % 24
        df["Minute"] =  (df[timestamp_column]  // minute) % 60
        # Convert seconds to datetime to extract month
        df['Datetime'] = pd.to_datetime(df[timestamp_column], unit='s', origin='unix')
        df['Month'] = df[datetime_column].dt.month

        # Cyclic encoding for hours, minutes, and months
        df[hour_features[0]] = np.sin(2 * np.pi * df["Hour"] / 24)
        df[hour_features[1]] = np.cos(2 * np.pi * df["Hour"] / 24)
        df[minute_features[0]] = np.sin(2 * np.pi * df["Minute"] / 60)
        df[minute_features[1]] = np.cos(2 * np.pi * df["Minute"] / 60)
        df[month_features[0]] = np.sin(2 * np.pi * df["Month"] / 12)
        df[month_features[1]] = np.cos(2 * np.pi * df["Month"] / 12)
        df = df.drop(drop_columns, axis=1)
        logging.info(f"printing df after adding cyclic features: {df.head()}")
        return df
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e 
 
def df_to_X_y(df: pd.DataFrame, time_steps:int=TIME_STEPS)-> tuple:
    """
    Converts a DataFrame into sequences (X) and corresponding labels (y) for time series prediction. 
    Args:
        df (pd.DataFrame): Input DataFrame to transform.
        time_steps (int): Number of time steps to include in each sequence. 
    Returns:
        tuple: A tuple containing:
            - X (np.array): Array of sequences.
            - y (np.array): Array of corresponding labels.
    """
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - time_steps): 
        row = df_as_np[i:i+time_steps]
        X.append(row) 
        label = df_as_np[i+time_steps][0] 
        y.append(label) 
    X = np.array(X)
    y = np.array(y)
     
    logging.info(f"Shapes - X: {X.shape}, y: {y.shape}")
    logging.info("First 2 samples of X:")
    logging.info(X[:2]) 
    logging.info("\nFirst 2 samples of y:")
    logging.info(y[:2]) 
    return X,y 

def scale_and_save_target(scaler, target_df: pd.DataFrame, scaler_file_path: str) -> np.array:
    """
    Scales the target features using a scaler, saves the scaler object, and returns the scaled data. 
    Args:
        scaler: Scaler object used for scaling.
        target_df (pd.DataFrame): DataFrame containing target features to scale.
        scaler_file_path (str): File path to save the scaler object. 
    Returns:
        np.array: Scaled target features.
    """
    try:
        logging.info("Entered the scale_and_save_target method of utils.")
         
        scaled_target = scaler.fit_transform(target_df)
         
        save_object(scaler_file_path, scaler) 
        logging.info(f"Scaler object saved successfully at {scaler_file_path}.")
        logging.info(f"Scaled target features shape: {scaled_target.shape}") 
        return scaled_target
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e
     
def prepare_batch_input(df: pd.DataFrame, time_steps: int=TIME_STEPS)-> np.array:
    """
    Prepares batch inputs by converting a DataFrame into sequences for time series prediction. 
    Args:
        df (pd.DataFrame): Input DataFrame.
        time_steps (int): Number of time steps to include in each batch sequence. 
    Returns:
        np.array: Array of batch sequences.
    """
    try:
        df_as_np = df.to_numpy() 
        dummy_column = np.zeros((df_as_np.shape[0], 1))
        df_as_np = np.hstack((dummy_column, df_as_np))
        
        X = []
        for i in range(len(df_as_np) - time_steps + 1): 
            row = df_as_np[i:i+time_steps]
            X.append(row)
        X = np.array(X)
        return X
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e