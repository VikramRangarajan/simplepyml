from simplepyml.util.functions.activation.activation import BaseActivation
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Relu(BaseActivation):
    r"""
    Rectified Linear Unit (ReLU) activation function.

    Definition:

    .. math::
        f(x) = 
        \begin{cases} 
        0 & x\leq 0 \\
        x & \text{otherwise}
        \end{cases}

    Derivative:

    .. math::
        f'(x) = 
        \begin{cases} 
        0 & x\leq 0 \\
        1 & \text{otherwise}
        \end{cases}

    Note:
        I have defined the discontinuity of the derivative at x = 0
        to be 0.
    """

    def __init__(self):
        ...

    def __call__(
        self, x: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.float64 | np.ndarray:
        """
        Return ReLU of input.

        Parameters
        ----------
        x : int | float | ndarray
            Input

        Returns
        -------
        ndarray
            ReLU of input
        """
        y = np.array(x, dtype=np.float64)
        y[y < 0] = 0
        return y

    def deriv(
        self, x: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.float64 | np.ndarray:
        """
        Derivative of ReLU.

        Notes
        -----
        Uses `np.heaviside`_ to calculate this step function

        .. _np.heaviside: https://numpy.org/doc/stable/reference/generated/numpy.heaviside.html

        Parameters
        ----------
        x : int | float | ndarray
            Input

        Returns
        -------
        ndarray
            Derivative of ReLU applied on input
        """
        return np.heaviside(x, 0).astype(np.float64)
