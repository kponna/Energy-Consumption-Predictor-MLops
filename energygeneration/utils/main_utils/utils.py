import yaml
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error,r2_score
from datetime import timedelta 
import os,sys
import numpy as np
import pandas as pd 
import pickle


def read_yaml_file(file_path:str) ->dict:
    try:
        with open(file_path,"rb") as yaml_file:
            # print(f"Schema file path: {file_path}")
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e
    

def write_yaml_file(file_path:str, content:object,replace:bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)

        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content,file)

    except Exception as e:
        raise EnergyGenerationException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.array):
    """
    Save the numpy array data to file.
    file_path(str): Location of the file to save.
    array(np.array): data to save.
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
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded"""
    try:
        with open(file_path,"rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e
    
def save_object(file_path:str,obj:object)-> None: 
    try:
        logging.info("Entered the save_object method of Main utils class")
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open (file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
        logging.info("Completed saving the scaling object.")
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e

def load_object(file_path:str,)-> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path,"rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise EnergyGenerationException(e,sys) from e
def add_cyclic_features(data: pd.DataFrame, schema_file: str) -> pd.DataFrame: 
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

        # Drop unnecessary columns
        df = df.drop(drop_columns, axis=1)
        logging.info(f"printing df after adding cyclic features: {df.head()}")
        return df
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e

 
def df_to_X_y(df, time_steps=12):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np) - time_steps):
        # Select the window of rows as input
        row = df_as_np[i:i+time_steps]
        X.append(row)
        # Select the target (value column)
        label = df_as_np[i+time_steps][0] 
        y.append(label)
    # Convert to NumPy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Log shapes and first 5 samples
    logging.info(f"Shapes - X: {X.shape}, y: {y.shape}")
    logging.info("First 2 samples of X:")
    logging.info(X[:2])  # Print first 5 samples of X
    logging.info("\nFirst 2 samples of y:")
    logging.info(y[:2])  # Print first 5 samples of y
    return X,y
  
def scale_and_save_target(scaler, target_df, scaler_file_path): 
    try:
        logging.info("Entered the scale_and_save_target method of utils.")
        
        # Fit the scaler to the target data
        scaled_target = scaler.fit_transform(target_df)
        
        # Save the scaler object
        save_object(scaler_file_path, scaler)
        
        logging.info(f"Scaler object saved successfully at {scaler_file_path}.")
        logging.info(f"Scaled target features shape: {scaled_target.shape}")
        
        return scaled_target
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e

# def evaluate_model(model, X_test, y_test, scaler,file_path): 
#     try:
#         predictions = model.predict(X_test).flatten()
#         predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
#         actuals_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

#         results_df = pd.DataFrame({
#             'Predictions': predictions_rescaled,
#             'Actuals': actuals_rescaled
#         })

#         mae = mean_absolute_error(actuals_rescaled, predictions_rescaled)
#         mse = mean_squared_error(actuals_rescaled, predictions_rescaled)
#         rmse = np.sqrt(mse)
#         r2 = r2_score(actuals_rescaled, predictions_rescaled)

#         metrics = {
#             'Mean Absolute Error (MAE)': mae,
#             'Mean Squared Error (MSE)': mse,
#             'Root Mean Squared Error (RMSE)': rmse,
#             'R² Score': r2
#         }

#         print("Evaluation Metrics:")
#         for metric, value in metrics.items():
#             print(f"{metric}: {value}")
#         return results_df
#     except Exception as e:
#         raise EnergyGenerationException(e, sys) from e  