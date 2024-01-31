from simplepyml.util.functions.activation.activation import BaseActivation
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Tanh(BaseActivation):
    r"""
    Hyperbolic tangent activation function.

    Definition:

    .. math::
        \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}

    Derivative:

    .. math::
        \tanh'(x) = sech^2(x) = \frac{1}{\cosh^2(x)}
    """

    def __init__(self):
        ...

    def __call__(
        self, x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        """
        Return tanh of input.

        Parameters
        ----------
        x : int | float | ndarray
            Input

        Returns
        -------
        ndarray
            tanh of input
        """
        return np.tanh(x, dtype=np.float64)

    def deriv(
        self, x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        """
        Derivative of tanh

        Parameters
        ----------
        x : int | float | ndarray
            Input

        Returns
        -------
        ndarray
            Derivative of tanh applied on input
        """
        return 1.0 / np.square(np.cosh(x, dtype=np.float64))
