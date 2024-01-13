from simplepyml.util.functions.loss.loss import BaseLoss
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class SCCE(BaseLoss):
    def __init__(self):
        ...

    def __call__(
        self,
        values: int | float | np.integer | np.floating | list | np.ndarray,
        expected: int | float | np.integer | np.floating | list | np.ndarray,
    ) -> np.ndarray | np.float64:
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        values = np.clip(values, a_min=0.001, a_max=0.999)
        expected = np.clip(expected, a_min=0.001, a_max=0.999)

        return -(
            (expected * np.log2(values) + (1 - expected) * np.log2(1 - values)).mean()
        )

    def deriv(
        self,
        values: np.ndarray | list,
        expected: np.ndarray | list,
    ) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        values = np.clip(values, a_min=0.001, a_max=0.999)
        expected = np.clip(expected, a_min=0.001, a_max=0.999)
        return -(
            expected / (values * np.log(2))
            - (1 - expected) / ((1 - values) * np.log(2))
        )
