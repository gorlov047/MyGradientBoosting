"""
The MyGB.exceptions` module includes all custom warnings and error
classes used across MyGB.
"""


class NotFittedError(ValueError, AttributeError):
    """Exception class to raise if estimator is used before fitting.

    This class inherits from both ValueError and AttributeError to help with
    exception handling.
    """


class UnidentifiedFeatureType(ValueError):
    """Exception class to raise if feature type is invalid.
    Valid feature types: `real` and `categorical`.
    """
