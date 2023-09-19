from typing import List, Tuple
from collections import Counter
from typing_extensions import Self
import numpy as np
from exceptions import NotFittedError, UnidentifiedFeatureType


class DecisionTreeRegressor():
    """Decision tree regressor with the mean squared error, which is equal to 
    variance reduction as feature selection criterion

    Parameters
    ----------
    feature_types: List[str]
        Valid feature types: `real` and `categorical`.

    max_depth : int, default=None
        The maximum depth of the tree.

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

        info_gain = -((left_count * left_var + right_count * right_var)
                      / feature_vector.shape[0])
        best_ind = np.argmax(info_gain)
        return tresholds, info_gain, tresholds[best_ind], info_gain[best_ind]

    def _fit_node(self, sub_x: np.ndarray, sub_y: np.ndarray, node: dict) -> None:
        if  self.min_samples_split and sub_y.shape[0] < self.min_samples_split:
            self._curr_depth -= 1
            return

        if np.all(np.isclose(sub_y, sub_y[0])):
            node["type"] = "terminal"
            node["value"] = sub_y[0]
            self._curr_depth -= 1
            return

        feature_best, threshold_best, gain_best, split = None, None, None, None
        for feature in range(sub_x.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}
            if feature_type == "real":
                feature_vector = sub_x[:, feature].astype(np.float64)
            elif feature_type == "categorical":
                counts = Counter(sub_x[:, feature])
                feature_vector = np.zeros(sub_x.shape[0])
                for key, current_count in counts.items():
                    mean = sub_y[sub_x[:, feature] == key].sum() / current_count
                    categories_map[key] = mean
                    feature_vector = np.where(sub_x[:, feature] == key, mean, feature_vector)
            else:
                raise UnidentifiedFeatureType

            if np.all(np.isclose(feature_vector, feature_vector[0])):
                continue

            _, _, threshold, gain = self._find_best_split(feature_vector, sub_y)
            if (gain_best is None or gain > gain_best):
                if self.min_samples_leaf:
                    n_sample_left = (feature_vector < threshold).sum()
                    if (n_sample_left < self.min_samples_leaf or
                        feature_vector.shape[0] - n_sample_left < self.min_samples_leaf):
                        continue
                feature_best = feature
                gain_best = gain
                split = feature_vector < threshold
                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(
                        map(lambda x: x[0],
                            filter(lambda x, t=threshold: x[1] < t, categories_map.items())
                        )
                    )
                else:
                    raise UnidentifiedFeatureType

        self._curr_depth += 1
        if feature_best is None or (self.max_depth and self._curr_depth > self.max_depth):
            node["type"] = "terminal"
            node["value"] = sub_y.mean()
            self._curr_depth -= 1
            return

        node["type"] = "nonterminal"
        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise UnidentifiedFeatureType

        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_x[split], sub_y[split], node["left_child"])
        self._fit_node(sub_x[np.logical_not(split)],
                       sub_y[np.logical_not(split)], node["right_child"])
        self._curr_depth -= 1

    def _predict_node(self, sample: np.ndarray, node: dict) -> float:
        if node["type"] == "terminal":
            return node["value"]
        f = node["feature_split"]
        if self._feature_types[f] == 'real':
            threshold = node["threshold"]
            if sample[f] < threshold:
                return self._predict_node(sample, node["left_child"])
            return self._predict_node(sample, node["right_child"])
        threshold = node["categories_split"]
        if sample[f] not in threshold:
            return self._predict_node(sample, node["right_child"])
        return self._predict_node(sample, node["left_child"])

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The training input samples.  

        y : array-like of shape (n_samples,)

        Returns
        -------
        self : DecisionTreeRegressor
            Fitted estimator.

        """

        self._fit_node(X, y, self._tree)
        self._fitted = True
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
            The predict values.
        """
        if not self._fitted:
            raise NotFittedError("Tree is not fitted")
        predicted = [self._predict_node(x, self._tree) for x in X]
        return np.array(predicted)
