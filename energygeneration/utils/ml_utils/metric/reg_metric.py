import sys
from energygeneration.exception_handling.exception import EnergyGenerationException 
from energygeneration.entity.artifact_entity import RegressionMetricArtifact
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 
import numpy as np

def get_model_score(y_true: np.array, y_pred: np.array) -> RegressionMetricArtifact:
    """
    Computes evaluation metrics for a regression model. 
    Args:
        y_true (np.array): The ground truth (actual) values.
        y_pred (np.array): The predicted values by the model. 
    Returns:
        RegressionMetricArtifact: An object containing the following regression metrics:
            - mae (float): Mean Absolute Error
            - mse (float): Mean Squared Error
            - rmse (float): Root Mean Squared Error
            - r2 (float): R-squared value indicating the goodness of fit.
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_true, y_pred)
        return RegressionMetricArtifact(mae=mae, mse=mse, rmse=rmse, r2=r2)
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e