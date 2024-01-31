from simplepyml.core.models.mlp.layers.layer import Layer
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# Dropout layer
class Dropout(Layer):
    """
    Dropout layer.

    Source: `Srivastava et al.`_

    .. _Srivastava et al.: https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf

    Parameters
    ----------
    Layer : _type_
        _description_
    """    
    def __init__(
        self,
        dropout: float,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dropout = dropout
        self.grad = dict()

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        # r will contain 0's and 1's, with approximately self.dropout % of 0's
        self.r = np.random.binomial(
            1.0,
            1 - self.dropout,
            input_array.shape,
        ).astype(np.float64)
        input_array *= self.r
        return input_array

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
        self.grad["input"] = dLda * self.r
