from simplepyml.core.models.mlp.layers.layer import Layer
from typing import Callable
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# Dense layer
class Dense(Layer):
    r"""
    Dense a.k.a. Fully Connected Layer.

    Definition:

    Takes an input, multiplies it with a weight matrix, adds a bias, and sends this result
    (z) through the activation function:

    .. math::
        \vec{z} &= W \cdot \vec{X} + \vec{b}

        \vec{Y} &= \phi(\vec{z})

    To calculate the loss gradient w.r.t. the biases, weights, and input, we use these formulas:

    .. math::
        \vec{\frac{\partial L}{\partial \vec b}} &= \vec{\frac{\partial L}{\partial \vec Y}} \odot \phi'(\vec z)

        \vec{\frac{\partial L}{\partial W}} &= \vec{\frac{\partial L}{\partial b}} \cdot \vec{X}^T

        \vec{\frac{\partial L}{\partial \vec X}} &= W^T \cdot \vec{\frac{\partial L}{\partial \vec b}}


    Parameters
    ----------
    size : int
        Output size.
    activation : function
        Activation function. See :py:mod:`~simplepyml.util.functions.activation` functions
    dropout : float
        Currently useless, future plans to implement dropout.

    Attributes
    ----------
    input_array : ndarray
        Most recent input of layer
    z : ndarray
        Output of layer before put through activation function
    dropout : float
        Currently useless.
    params : dict()
        Contains weight and bias arrays
    grad : dict()
        Empty dictionary, until :py:func:`~back_grad` is called, after which:

        - grad["input"] is the loss gradient w.r.t. the input
        - grad["weights"] is the loss gradient w.r.t. the input
        - grad["biases"] is the loss gradient w.r.t. the biases
    param_num : int
        Number of parameters in this layer (params["weights"].size + params["biases"].size)
    initialized : bool
        Whether the layer has been initialized. False until called for the
        first time
    """

    def __init__(
        self,
        size: int | np.integer,
        activation: Callable[[np.ndarray], np.ndarray],
        dropout: float | np.floating = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        size = int(size)
        # TODO: dropout
        dropout = float(dropout)

        if size < 1:
            raise ValueError("Size of dense layer must be >= 1")

        self.activation_func = activation

        self.size = size

        self.grad = dict()

        self.dropout = dropout

    def _init_layer(self, input_array):
        if input_array.ndim > 1:
            raise Exception("Input must be 1-Dimensional for Dense Layer")
        self.initialized = True
        input_size = input_array.size
        self.params["weights"] = np.random.uniform(
            size=(self.size, input_size), low=-1, high=1
        )
        self.params["biases"] = np.random.uniform(size=self.size, low=-1, high=1)
        self.param_num = self.params["weights"].size + self.params["biases"].size

    def __call__(self, input_array: np.ndarray):
        if not self.initialized:
            self._init_layer(input_array)
        self.input_array = input_array
        self.z = self.params["weights"] @ input_array + self.params["biases"]
        return self.activation_func(self.z)

    def back_grad(self, dLda: np.ndarray):
        """
        Backward gradient calculation. Given loss gradient w.r.t. the most
        recent output, calculate and store the loss gradient w.r.t. the most
        recent input, weights, and biases and stores in the grad dictionary.

        Parameters
        ----------
        dLda : ndarray
            The loss gradient w.r.t. the most recent output. Must have shape
            equal to the output shape

        Returns
        -------
        None
        """
        phi_prime_z = self.activation_func.deriv(self.z)
        self.grad["biases"] = np.multiply(dLda, phi_prime_z)
        self.grad["input"] = self.params["weights"].T @ self.grad["biases"]
        self.grad["weights"] = np.reshape(self.grad["biases"], (-1, 1)) @ np.reshape(
            self.input_array, (1, -1)
        )
