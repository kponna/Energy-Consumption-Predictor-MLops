import sys
from energygeneration.exception_handling.exception import EnergyGenerationException
from energygeneration.logging.logger import logging
from energygeneration.entity.artifact_entity import RegressionMetricArtifact
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score 
import numpy as np
def get_model_score(y_true, y_pred) -> RegressionMetricArtifact:
    """
    Get evaluation metrics for a regression model.
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse) 
        r2 = r2_score(y_true, y_pred)

        return RegressionMetricArtifact(mae=mae, mse=mse, rmse=rmse, r2=r2)
    except Exception as e:
        raise EnergyGenerationException(e, sys) from e