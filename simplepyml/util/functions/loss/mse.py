import numpy as np
from .loss import BaseLoss

class MSE(BaseLoss):
    def __init__(self):
        ...
    
    def __call__(
        self, 
        values: int | float | np.integer | np.floating | list | np.ndarray, 
        expected: int | float | np.integer | np.floating | list | np.ndarray
    ) -> np.ndarray | np.float64:
        if isinstance(values, list):
            values = np.array(values)
        if isinstance(expected, list):
            expected = np.array(expected)
        
        return np.mean((expected - values)**2, dtype=np.float64)
    
    def deriv(
        self, 
        values: np.ndarray | list, 
        expected: np.ndarray | list,
    ) -> np.ndarray:
        values = np.array(values)
        expected = np.array(expected)
        return 2.0 / values.size * (values - expected)