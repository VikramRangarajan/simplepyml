from .activation import BaseActivation
import numpy as np

class Sigmoid(BaseActivation):
    def __init__(self):
        ...
    
    def __call__(
        self,
        x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        return 1.0/(1+np.exp(-x, dtype=np.float64))
    
    def deriv(
        self,
        x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        sx = self(x)
        return sx*(1-sx)