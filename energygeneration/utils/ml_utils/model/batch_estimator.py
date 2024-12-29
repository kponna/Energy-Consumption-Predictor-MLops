import sys  
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging 

class EnergyModel:
    def __init__(self,model): 
        try:
            self.model = model
        except Exception as e:
            raise EnergyGenerationException(e, sys) 
    
    def predictor(self, X):
        try:
            if self.model is None:
                raise ValueError("Model is not built. Call `build_model` and train before prediction.")
            
            logging.info("Generating predictions using the LSTM model.") 
            y_pred = self.model.predict(X).flatten()
            return y_pred
        except Exception as e:
            raise EnergyGenerationException(e, sys) 