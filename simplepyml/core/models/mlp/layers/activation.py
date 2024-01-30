from simplepyml.core.models.mlp.layers.layer import Layer
from typing import Callable
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# Activation layer
class Activation(Layer):
    r"""
    Activation Layer.

    Definition:

    Takes an input, applies an activation function to it.

    .. math::
        Y = \phi(X)

    To calculate the loss gradient w.r.t. the input, we use this formula:

    .. math::
        \frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} \odot \phi'(z)


    Parameters
    ----------
    activation : function
        Activation function. See :py:mod:`~simplepyml.util.functions.activation` functions

    Attributes
    ----------
    input_array : ndarray
        Most recent input of layer
    params : dict()
        Empty dictionary, no parameters in this layer
    grad : dict()
        Empty dictionary, until :py:func:`~back_grad` is called, after which:

        - grad["input"] is the loss gradient w.r.t. the input
    param_num : int (0)
        Number of parameters in this layer (0)
    initialized : bool
        Whether the layer has been initialized. False until called for the
        first time
    """

    def __init__(
        self,
        activation: Callable[[np.ndarray], np.ndarray],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.activation = activation
        self.grad = dict()

    def _init_layer(self, input_array):
        self.initialized = True
        self.input_array = input_array

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self._init_layer(input_array)
        return self.activation(input_array)

    def back_grad(self, dLda: np.ndarray) -> None:
        """
        Backward gradient calculation. Given loss gradient w.r.t. the most
        recent output, calculate and store the loss gradient w.r.t. the most
        recent input and stores in the grad dictionary.

        Parameters
        ----------
        dLda : ndarray
            The loss gradient w.r.t. the most recent output. Must have shape
            equal to the output shape

        Returns
        -------
        None
        """
        self.grad["input"] = dLda * self.activation.deriv(self.input_array)
