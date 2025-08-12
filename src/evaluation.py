from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Any, Dict
import numpy as np



class EvaluateTask:
    def __init__(self, y_true: Any, y_pred: Any):
        """
        Initializes the evaluation task with true and predicted values.

        Args:
            y_true (array-like): True target values.
            y_pred (array-like): Predicted target values.
        """
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)

        if self.y_true.shape != self.y_pred.shape:
            raise ValueError("y_true and y_pred must have the same shape.")

    def evaluate_model(self) -> Dict[str, float]:
        """
        Evaluates the model performance using various metrics.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        mae = mean_absolute_error(self.y_true, self.y_pred)
        mse = mean_squared_error(self.y_true, self.y_pred)
        rmse = mse ** 0.5
        r2 = r2_score(self.y_true, self.y_pred)

        return {
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        }  
    
     
        
