import numpy as np

# Any possible activation functions needed for layers



def activation_function_from_str(x: str):
    if x.lower() == "sigmoid":
        return sigmoid
    if x.lower() == "relu":
        return relu
    
    raise ValueError("Invalid activation function")

# Returns the derivative of the input activation function. Accepts function pointers.
# Ex: deriv(sigmoid) = sigmoid_deriv
def deriv(f: function):
    if f == sigmoid:
        return sigmoid_deriv
    if f == relu:
        return relu_deriv
    
    raise ValueError("Invalid activation function")

# Sigmoid logistic curve
def sigmoid(
    x: int | float | np.integer | np.floating | list
) -> np.number:
    if isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, np.array):
        return [1.0/(1+np.exp(-x_i)) for x_i in x]
    return 1.0/(1+np.exp(-x))

# Derivative of sigmoid function. Returns sigmoid(x)*(1-sigmoid(x))
def sigmoid_deriv(
    x: int | float | np.integer | np.floating | list
) -> np.number:
    if isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, np.array):
        return [sigmoid(x_i)*(1-sigmoid(x_i)) for x_i in x]
    return sigmoid(x)*(1-sigmoid(x))


# Rectified Linear Unit
def relu(
    x: int | float | np.integer | np.floating | list
) -> np.number:
    
    if isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, np.array):
        return [np.float64(x_i) if x_i > 0 else 0 for x_i in x]
    
    return np.float64(x) if x > 0 else 0

# Derivative of RELU: 1 if x>0, 0 if x < 0. relu_deriv(0) = 1, doesn't really matter.
def relu_deriv(
    x: int | float | np.integer | np.floating | list
) -> np.number:
    if isinstance(x, list) or isinstance(x, np.ndarray) or isinstance(x, np.array):
        return [np.float64(1.0) if x_i >= 0 else np.float64(0) for x_i in x]
    return np.float64(1.0) if x >= 0 else np.float64(0)