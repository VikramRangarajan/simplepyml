from .activation import BaseActivation
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
        return np.ones(shape=x.shape)
