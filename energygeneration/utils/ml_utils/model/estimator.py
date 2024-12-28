import sys 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint 
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging

class EnergyLstmModel:
    def __init__(self, input_shape, learning_rate=0.0001, model_checkpoint_path=None):  
        try:
            self.input_shape = input_shape
            self.learning_rate = learning_rate
            self.model_checkpoint_path = model_checkpoint_path
            self.model = None
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def build_model(self): 
        try:
            model = Sequential()
            model.add(InputLayer((12, 11)))
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
                optimizer=Adam(learning_rate=self.learning_rate),
                metrics=[RootMeanSquaredError()]
            )   
            self.model = model
            logging.info("LSTM model built successfully.")
        except Exception as e:
            raise EnergyGenerationException(e, sys)

    def train_model(self, X_train, y_train, X_val, y_val, epochs=50): 
        try:
            if self.model is None:
                raise ValueError("Model is not built. Call `build_model` before training.")

            callbacks = []
            if self.model_checkpoint_path:
                callbacks.append(ModelCheckpoint(self.model_checkpoint_path, save_best_only=True))

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

    def predict(self, X): 
        try:
            if self.model is None:
                raise ValueError("Model is not built. Call `build_model` and train before prediction.")
            
            logging.info("Generating predictions using the LSTM model.")
             
            y_pred = self.model.predict(X).flatten()
             
            logging.info("Predictions generated and rescaled successfully.")
            return y_pred
        except Exception as e:
            raise EnergyGenerationException(e, sys) 