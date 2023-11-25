from .activation import BaseActivation
import numpy as np

class Relu(BaseActivation):
    def __init__(self):
        ...
    
    def __call__(
        self,
        x: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.float64 | np.ndarray:
        func = lambda x: np.float64(x) if x > 0 else np.float64(0)
        func = np.vectorize(func)
        if isinstance(x, list):
            x = np.array(x)
        return func(x)
    
    def deriv(
        self,
        x: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.float64 | np.ndarray:
        func = lambda x: np.float64(1) if x > 0 else np.float64(0)
        func = np.vectorize(func)
        if isinstance(x, list):
            x = np.array(x)
        
        return func(x)