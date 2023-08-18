import numpy as np
# All loss functions required for Neural Networks

# Mean Squared Error
def mse(values, expected):
    if isinstance(values, list):
        values = np.array(values)
    if isinstance(expected, list):
        expected = np.array(expected)
    
    return 1/len(values) * np.sum((expected - values)**2)

def mse_deriv(value, expected):
    return 2.0 / len(value) * (value - expected)