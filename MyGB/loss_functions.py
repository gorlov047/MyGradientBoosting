import numpy as np

class LossFunction():
    """Base class for all loss functions."""
    def calculate(self, y_true: np.ndarray, y_pred:np.ndarray) -> np.number:
        "calculate loss"

    def calc_gradient(self, y_true: np.ndarray, y_pred:np.ndarray) -> np.ndarray:
        "calculate loss gradient"


class MSE(LossFunction):
    """Mean Squared Error class."""

    def calculate(self, y_true, y_pred):
        return ((y_true - y_pred) ** 2).mean()

    def calc_gradient(self, y_true, y_pred):
        return -2 * y_pred * (y_true - y_pred)


class LogLoss(LossFunction):
    """Log loss as a function of margin."""

    def __init__(self):
        super().__init__()
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))

    def calculate(self, y_true, y_pred):
        y_proba = self.sigmoid(y_pred)
        return -(y_true * np.log(y_proba) + (1 - y_true) * np.log(1 - y_proba)).mean()

    def calc_gradient(self, y_true, y_pred):
        y_proba = self.sigmoid(y_pred)
        return (1 - y_true) * y_proba - y_true * (1 - y_proba)
