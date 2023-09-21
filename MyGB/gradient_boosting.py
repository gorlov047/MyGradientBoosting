from typing_extensions import Self
import numpy as np
from tree_regressor import DecisionTreeRegressor
from loss_functions import LossFunction


class GradientBoosting:
    """Gradient Boosting on Decision Trees class.

    Parameters
    ----------
    base_model_params: dict, default=None
        Parameters for decision tree.
    n_estimators: int, default=10
        The number of trees in the ensemble.
    learning_rate: float, default=0.1
        Learning rate like in normal GD 
    subsample_size: float, default=0.3
        n_samples * subsample_size will be using for validation
        if early_stopping_rounds is not None
    loss: LossFunction, default=None
        The loss function that we will optimize
    early_stopping_rounds: int, default=None)
        The number of consecutive rounds in which the validation
        error worsened before stopping the training
    """
    def __init__(
            self,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample_size: float = 0.3,
            loss: LossFunction = None,
            early_stopping_rounds: int = None):
        self.base_model_class = DecisionTreeRegressor
        self.base_model_params: dict = {} if base_model_params is None else base_model_params
        self.n_estimators: int = n_estimators
        self.models: list = []
        self.gammas: list = []
        self.learning_rate: float = learning_rate
        self.subsample_size: float = subsample_size
        self.loss_fn = loss

        self.early_stopping_rounds: int = early_stopping_rounds
        if early_stopping_rounds is not None:
            self.round_counter = 0

        self.train_history = []
        self.val_history = []

    def _find_optimal_gamma(self, y, old_predictions, new_predictions) -> float:
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def _fit_new_base_model(self, X: np.ndarray, y: np.ndarray, y_pred: np.ndarray) -> None:
        subsample_inds = np.random.choice(X.shape[0],
                                          int(X.shape[0] * self.subsample),
                                          replace=True)
        model = self.base_model_class(**self.base_model_params)
        model.fit(X[subsample_inds],
                   -self.loss.calc_gradient(y[subsample_inds], y_pred[subsample_inds]))
        gamma = self.find_optimal_gamma(y, y_pred, model.predict(X))
        self.gammas.append(gamma * self.learning_rate)
        self.models.append(model)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
             x_valid: np.ndarray, y_valid: np.ndarray) -> Self:
        """Build a gradient boosting on decision tree regressor from the training set (X, y).

        Parameters
        ----------
        x_train: array like of shape (n_samples * subsample_size, n_features)
            The training input samples.
        y_train: array like of shape (n_samples * subsample_size,)
            Targets for training samples.
        x_valid: array like of shape (n_samples * (1 - subsample_size), n_features)
            The validation input samples to avoid overfitting if
            early_stopping_rounds parameter is set.
        y_valid: array like of shape (n_samples * (1 - subsample_size),)
            Validation targets
        
        Returns
        -------
        self : GradientBoosting
            Fitted estimator.
        """
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        self.train_history.append(self.loss_fn.calculate(y_train, train_predictions))
        self.val_history.append(self.loss_fn.calculate(y_valid, valid_predictions))

        for _ in range(self.n_estimators):
            self._fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions = self.predict(x_train)
            new_val_preds = self.predict(x_valid)
            if self.early_stopping_rounds is not None:
                if (self.loss_fn(y_valid, new_val_preds) >=
                        self.loss_fn(y_valid, valid_predictions)):
                    self.round_counter += 1
                    if self.round_counter == self.early_stopping_rounds:
                        return self
                else:
                    self.round_counter = 0

            valid_predictions = new_val_preds
            self.train_history.append(self.loss_fn.calculate(y_train, train_predictions))
            self.val_history.append(self.loss_fn.calculate(y_valid, valid_predictions))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target vector for X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.
        
        Returns
        -------
        y : array-like of shape (n_samples,)
        """
        preds = np.zeros(X.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            preds += gamma * model.predict(X)
        return preds

    