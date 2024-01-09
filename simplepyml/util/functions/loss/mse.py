import numpy as np
from simplepyml.util.functions.loss.loss import BaseLoss


class MSE(BaseLoss):
    def __init__(self):
        ...

    def __call__(
        self,
        values: int | float | np.integer | np.floating | list | np.ndarray,
        expected: int | float | np.integer | np.floating | list | np.ndarray,
    ) -> np.ndarray | np.float64:
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)

        return np.mean((expected - values) ** 2, dtype=np.float64)

    def deriv(
        self,
        values: np.ndarray | list,
        expected: np.ndarray | list,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        return 2.0 / values.size * (values - expected)
