from simplepyml.core.models.mlp.layers.layer import Layer
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# Reshape layer
class Reshape(Layer):
    r"""
    Reshape Layer.

    Definition:

    Takes an input, reshapes it.

    To calculate the loss gradient w.r.t. the input, we just reshape the output
    shape into the input shape.

    Parameters:
    -----------
    output_shape : tuple
        The desired output shape. The size of the output array must be the same
        as the input array

    Attributes:
    -----------
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
        output_shape: tuple,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.output_shape = output_shape
        self.grad = dict()

    def _init_layer(self, input_array):
        self.initialized = True
        self.input_shape = input_array.shape

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self._init_layer(input_array)
        return np.reshape(input_array, self.output_shape)

    def back_grad(self, dLda: np.ndarray):
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
        self.grad["input"] = np.reshape(dLda, self.input_shape)


# Returns a layer that flattens its input
def Flatten():
    """
    Gives you a :py:class:`~Reshape` object which reshapes a given input
    to a 1d array

    Returns
    -------
    None
    """
    return Reshape((-1,))
