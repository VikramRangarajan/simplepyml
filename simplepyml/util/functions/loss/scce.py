from simplepyml.util.functions.loss.loss import BaseLoss
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class SCCE(BaseLoss):
    r"""
    Categorical Cross Entropy Class (CCE). Sparse by default,
    need to add non-sparse in the future.

    Definition:

    .. math::
        L(\hat Y, Y) = -\sum_{i=1}^n Y \log_{b}(\hat Y)

    Derivative:

    .. math::
        \frac{\partial L}{\partial \hat Y} = -\frac{1}{\ln(b)} \frac{Y}{\hat Y}
    """

    def __init__(self):
        ...

    def __call__(
        self,
        values: int | float | np.integer | np.floating | list | np.ndarray,
        expected: int | float | np.integer | np.floating | list | np.ndarray,
        log_base: float = 2,
    ) -> np.ndarray | np.float64:
        """
        Returns the CCE result of the expected and output values.

        Parameters
        ----------
        values : int | float | np.integer | np.floating | list | np.ndarray
            output values
        expected : int | float | np.integer | np.floating | list | np.ndarray
            expected or correct values
        log_base : float, optional
            Log base used (default 2)

        Returns
        -------
        np.ndarray | np.float64
            CCE result
        """
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        return -np.sum(expected * np.log2(log_base, values) / np.log2(log_base))

    def deriv(
        self,
        values: np.ndarray | list,
        expected: np.ndarray | list,
        log_base: float = 2,
    ) -> np.ndarray:
        """
        Given the loss gradient w.r.t. the output, calculates the loss
        gradient w.r.t. the input and stores in the grad dictionary

        Parameters
        ----------
        values : ndarray
            Output values
        expected : np.ndarray | list
            Expected values
        log_base : float, optional
            Log base used (default 2)

        Returns
        -------
        ndarray
            Gradient array w.r.t. input
        """
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        return (-1.0 / np.log(log_base)) * (expected / (values))
