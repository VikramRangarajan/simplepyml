from .activation import BaseActivation
import numpy as np

class Linear(BaseActivation):
    def __init__(self):
        ...
    
    def __call__(self, x):
        return x
    
    def deriv(
        self,
        x: int | float | np.integer | np.floating | list
    ) -> np.float64 | np.ndarray:
        if isinstance(x, list):
            x = np.array(x)
        
        if isinstance(x, np.ndarray):
            return np.ones(shape=x.shape)
        
        return 1