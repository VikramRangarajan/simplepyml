from simplepyml.util.functions.activation.activation import BaseActivation
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Sigmoid(BaseActivation):
    r"""
    Sigmoid activation function.

    Definition:

    .. math::
        \sigma(x) = \frac{1}{1 + e^{-x}}

    Derivative:

    .. math::
        \sigma'(x) = \sigma(x)(1-\sigma(x))
    """

    def __init__(self):
        ...

    def __call__(
        self, x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        """
        Return sigmoid of input.

        Parameters
        ----------
        x : int | float | ndarray
            Input

        Returns
        -------
        ndarray
            sigmoid of input
        """
        return 1.0 / (1 + np.exp(-x, dtype=np.float64))

    def deriv(
        self, x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        """
        Derivative of sigmoid

        Parameters
        ----------
        x : int | float | ndarray
            Input

        Returns
        -------
        ndarray
            Derivative of sigmoid applied on input
        """
        sx = self(x)
        return sx * (1 - sx)
