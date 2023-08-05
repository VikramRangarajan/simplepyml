import numpy as np

# Any possible activation functions needed for layers



def activation_function_from_str(x: str):
    if x.lower() == "sigmoid":
        return sigmoid
    if x.lower() == "relu":
        return relu
    
    raise ValueError("Invalid activation function")

def sigmoid(
    x: int | float | np.integer | np.floating | list
) -> np.number:
    if isinstance(x, list):
        return [1.0/(1+np.exp(-x_i)) for x_i in x]
    return 1.0/(1+np.exp(-x))


def relu(
    x: int | float | np.integer | np.floating | list
) -> np.number:
    
    if isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, np.array):
        return [np.float64(x_i) if x_i > 0 else 0 for x_i in x]
    
    return np.float64(x) if x > 0 else 0