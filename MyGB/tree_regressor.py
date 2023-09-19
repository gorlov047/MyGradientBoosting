from typing import List, Tuple
import numpy as np


class DecisionTreeRegressor():
    """Decision _tree regressor with the mean squared error, which is equal to 
    variance reduction as feature selection criterion

    Parameters
    ----------
    feature_types: List[str]
        Valid feature types: `real` and `categorical`.

    max_depth : int, default=None
        The maximum depth of the _tree.

    min_samples_split : int, default=None
        The minimum number of samples required to split an internal node:

    min_samples_leaf : int, default=None
        The minimum number of samples required to be at a leaf node.
    """

    def __init__(self,
                 feature_types: List[str],
                 max_depth: int = None,
                 min_samples_split: int = None,
                 min_samples_leaf: int = None):
        self._tree = {}
        self._fitted = False
        self._feature_types = feature_types
        self.max_depth = max_depth
        self._curr_depth = 0
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf


    def _find_best_split(self,
                         feature_vector: np.ndarray,
                         target_vector:np.ndarray
                         )-> Tuple[np.ndarray, np.ndarray, np.number, np.number]:
        pass

    def _fit_node(self, sub_x: np.ndarray, sub_y: np.ndarray, node: dict) -> None:
        pass
        
    def _predict_node(self, sample: np.ndarray, node: dict) -> float:
        pass
 
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """fit model"""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """predict targets for the samples"""
        pass