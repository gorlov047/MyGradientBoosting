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
        def variance(sum_, sum_of_squares, number):
            return sum_of_squares / number - (sum_ / number) ** 2

        sorted_inds = np.argsort(feature_vector)
        sorted_feature, inds = np.unique(feature_vector[sorted_inds],
                                         return_index=True)
        tresholds = (sorted_feature[1:] + sorted_feature[:-1]) / 2
        left_sum = np.cumsum(target_vector[sorted_inds])[inds - 1][1:]
        left_sq_sum = np.cumsum(target_vector[sorted_inds] ** 2)[inds - 1][1:]
        left_count = np.arange(1, feature_vector.shape[0], 1)[inds - 1][1:]
        left_var = variance(left_sum, left_sq_sum, left_count)

        right_count = feature_vector.shape[0] - left_count
        right_var = variance(target_vector.sum() - left_sum,
                             (target_vector ** 2).sum() - left_sq_sum,
                             right_count)

        info_gain = -((left_count * left_var + right_count * right_var) / feature_vector.shape[0])
        best_ind = np.argmax(info_gain)
        return tresholds, info_gain, tresholds[best_ind], info_gain[best_ind]

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
