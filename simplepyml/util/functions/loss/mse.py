from simplepyml.util.functions.loss.loss import BaseLoss
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class MSE(BaseLoss):
    r"""
    Mean Square Error Loss Function.

    Uses this formula:

    .. math::
        L=\frac{1}{n} \sum_{i=1}^n (Y_i - \hat Y_i)^2
    """    
    def __init__(self):
        ...

    def __call__(
        self,
        values: int | float | np.integer | np.floating | list | np.ndarray,
        expected: int | float | np.integer | np.floating | list | np.ndarray,
    ) -> np.ndarray | np.float64:
        """
        Returns the MSE result of the expected and output values.

        Parameters
        ----------
        values : int | float | list | ndarray
            Output values
        expected : int | float | list | ndarray
            Expected/correct values

        Returns
        -------
        ndarray | float64
            MSE result
        """        
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)

        return np.mean((expected - values) ** 2, dtype=np.float64)

    def deriv(
        self,
        values: np.ndarray | list,
        expected: np.ndarray | list,
    ) -> np.ndarray:
        r"""
        Derivative of MSE w.r.t. output values, using this formula:

        .. math::
            \frac{\partial L}{\partial \hat Y} = \frac{2}{n} (\hat{\vec{Y}} - \vec Y)

        Parameters
        ----------
        values : list | ndarray
            Output values
        expected : list | ndarray
            Expected/correct values

        Returns
        -------
        ndarray
            :math:`\frac{\partial L}{\partial \hat Y}`
        """        
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        return 2.0 / values.size * (values - expected)
