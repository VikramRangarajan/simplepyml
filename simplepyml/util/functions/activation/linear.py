from simplepyml.util.functions.activation.activation import BaseActivation
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Linear(BaseActivation):
    def __init__(self):
        ...

    def __call__(self, x):
        return x

    def deriv(
        self, x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        x = np.asarray(x)
        return np.ones(shape=x.shape, dtype=np.float64)
