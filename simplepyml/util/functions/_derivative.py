from .activation import (
    sigmoid,
    sigmoid_deriv,
    relu,
    relu_deriv,
    linear,
    linear_deriv,
)
from .loss import (
    mse,
    mse_deriv,
    scce,
    scce_deriv,
)
# Returns the derivative of the input activation function. Accepts function pointers.
# Ex: deriv(sigmoid) = sigmoid_deriv
def deriv(f):
    if f == sigmoid:
        return sigmoid_deriv
    if f == relu:
        return relu_deriv
    if f == mse:
        return mse_deriv
    if f == linear:
        return linear_deriv
    if f == scce:
        return scce_deriv
    
    raise ValueError("Invalid activation function")