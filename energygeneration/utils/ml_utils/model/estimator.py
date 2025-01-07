import sys
import numpy as np
from energygeneration.logging.logger import logging
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.constant import TIME_STEPS,LEARNING_RATE,EPOCHS,PATIENCE

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError 
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class EnergyLstmModel:
    """
    A class to build, train, and use an LSTM-based energy prediction model. 
    """
    def __init__(self, model_checkpoint_path=None):
        """
        Initialize the EnergyLstmModel. 
        Args:
            model_checkpoint_path (str, optional): Path to save the best model. Defaults to None. 
        """
        try:  
            self.model_checkpoint_path = model_checkpoint_path
            self.model = None
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def build_model(self):
        """
        Build the LSTM model with a specified architecture.
        """
        try:
            model = Sequential()
            model.add(InputLayer((TIME_STEPS, 11)))
            model.add(LSTM(64))
            model.add(Dense(128))
            model.add(Dense(64))
            model.add(Dense(64))
            model.add(Dense(64))
            model.add(Dropout(0.3))
            model.add(Dense(64))
            model.add(Dropout(0.3))
            model.add(Dense(16))
            model.add(Dense(8, 'relu'))
            model.add(Dense(1, 'linear')) 
            model.summary()
            model.compile(
                loss=MeanSquaredError(),
                optimizer=Adam(learning_rate=LEARNING_RATE),
                metrics=[RootMeanSquaredError()]
            )   
            self.model = model
            logging.info("LSTM model built successfully.")
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=EPOCHS):
        """
        Train the LSTM model. 
        Args:
            X_train (np.array): Training features.
            y_train (np.array): Training labels.
            X_val (np.array): Validation features.
            y_val (np.array): Validation labels.
            epochs (int): Number of training epochs. 
        Returns:
            History: Training history object. 
        """
        try:
            if self.model is None:
                raise ValueError("Model is not built. Call `build_model` before training.")

            callbacks = []
            if self.model_checkpoint_path:
                callbacks.append(ModelCheckpoint(self.model_checkpoint_path, save_best_only=True))

            # Add EarlyStopping callback
            early_stopping = EarlyStopping(
                monitor='val_loss',  
                patience=PATIENCE, 
                restore_best_weights=True 
            )
            callbacks.append(early_stopping)
            
            logging.info("Starting LSTM model training...")
            history = self.model.fit(
                X_train,
                y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                callbacks=callbacks,
                verbose=1
            ) 
            logging.info("LSTM model training completed.")
            return history
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def predict(self, X)-> np.array:
        """
        Generate predictions using the trained LSTM model. 
        Args:
            X (np.array): Input data for prediction. 
        Returns:
            np.array: Predicted values as a flattened array. 
        """
        try:
            if self.model is None:
                raise ValueError("Model is not built. Call `build_model` and train before prediction.")
            
            logging.info("Generating predictions using the LSTM model.")
             
            y_pred = self.model.predict(X).flatten()
             
            logging.info("Predictions generated and rescaled successfully.")
            return y_pred
        except Exception as e:
            raise EnergyGenerationException(e, sys)