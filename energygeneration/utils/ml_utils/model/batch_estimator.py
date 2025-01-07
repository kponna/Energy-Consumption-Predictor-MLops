import sys 
from energygeneration.logging.logger import logging 
from energygeneration.exception_handling.exception import EnergyGenerationException

class EnergyModel:
    """
    A class representing the energy prediction model using LSTM. 
    """
    def __init__(self,model):
        """
        Initializes the EnergyModel with a given model. 
        Args:
            model: A trained LSTM model instance. 
        """
        try:
            self.model = model
        except Exception as e:
            raise EnergyGenerationException(e, sys) 
    
    def predictor(self, X):
        """
        Generates predictions using the trained model. 
        Args:
            X (np.array): Input data for prediction, shaped appropriately for the model. 
        Returns:
            np.array: Predicted values as a flattened array. 
        """
        try:
            if self.model is None:
                raise ValueError("Model is not built. Call `build_model` and train before prediction.") 
            logging.info("Generating predictions using the LSTM model.") 
            y_pred = self.model.predict(X).flatten()
            return y_pred
        except Exception as e:
            raise EnergyGenerationException(e, sys) 