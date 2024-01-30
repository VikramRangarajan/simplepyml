from simplepyml.util.functions.activation.activation import BaseActivation
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Linear(BaseActivation):
    """
    Linear Activation Function.

    f(x) = x (scalar or array)
    f'(x) = 1 (scalar or array of ones)
    """

    def __init__(self):
        ...

    def __call__(self, x):
        """
        Returns input

        Parameters
        ----------
        x : object
            Input

        Returns
        -------
        x
            Input
        """
        return x

    def deriv(
        self, x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        """
        Returns derivative of x, which is one

        Parameters
        ----------
        x : object
            Input

        Returns
        -------
        ndarray
            ndarray filled with ones, same shape as input
        """
        x = np.asarray(x)
        return np.ones(shape=x.shape, dtype=np.float64)
