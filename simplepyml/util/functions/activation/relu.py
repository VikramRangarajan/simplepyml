from simplepyml.util.functions.activation.activation import BaseActivation
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Relu(BaseActivation):
    def __init__(self):
        ...

    def __call__(
        self, x: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.float64 | np.ndarray:
        y = np.array(x, dtype=np.float64)
        y[y < 0] = 0
        return y

    def deriv(
        self, x: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.float64 | np.ndarray:
        y = np.array(x, dtype=np.float64)
        y[y <= 0] = 0
        y[y > 0] = 1
        return y
