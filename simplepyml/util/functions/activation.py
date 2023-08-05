import numpy as np

# Any possible activation functions needed for layers



def activation_function_from_str(x: str):
    if x.lower() == "sigmoid":
        return sigmoid
    if x.lower() == "relu":
        return relu

def sigmoid(
    x: int | float | np.integer | np.floating
) -> np.number:

    return 1.0/(1+np.exp(-x))


def relu(
    x: int | float | np.integer | np.floating
) -> np.number:
    
    if not isinstance(x, np.floating) and not isinstance(x, np.integer):
        # If x is s a python native data type, convert to numpy dtype
        return np.float64(x) if x > 0 else 0
    
    return x if x > 0 else 0