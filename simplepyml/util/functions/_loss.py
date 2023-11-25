import numpy as np
# All loss functions required for Neural Networks

# Mean Squared Error
def mse(values, expected):
    if isinstance(values, list):
        values = np.array(values)
    if isinstance(expected, list):
        expected = np.array(expected)
    
    return np.mean((expected - values)**2)

def mse_deriv(value, expected):
    return 2.0 / len(value) * (value - expected)

# Sparse Categorical Cross Entropy
def scce(values, expected):
    values = np.clip(values, a_min=0.001, a_max=0.999)
    expected = np.clip(expected, a_min=0.001, a_max=0.999)

    return -((expected*np.log2(values) + (1-expected)*np.log2(1-values)).mean())

def scce_deriv(values, expected):
    values = np.clip(values, a_min=0.001, a_max=0.999)
    expected = np.clip(expected, a_min=0.001, a_max=0.999)
    return -((expected/(values*np.log(2)) - (1-expected)/((1-values)*np.log(2))))