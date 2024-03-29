from simplepyml.core.models.mlp.layers.layer import Layer
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# Activation layer
class Softmax(Layer):
    r"""
    Softmax Activation Layer.

    Definition:

    .. math::
        Y_i = \frac{e^{X_i}}{\sum_{k=1}^{n} e^{X_k}} \text{ for all i}

    Derivative of loss w.r.t. input, given :math:`\frac{\partial L}{\partial Y}`:

    .. math::
        g &= \frac{\partial L}{\partial Y} \odot Y

        \frac{\partial L}{\partial X} &= (-\sum_{i=1}^n g_i) Y + g


    My derivation of these formulas is here: :download:`pdf <../pdfs/softmaxlayer_proof.pdf>`
    """

    def __init__(
        self,
        dropout: float | None = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.grad = dict()
        self.dropout = dropout

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        """
        Returns the softmax result of the input

        Parameters
        ----------
        input_array : ndarray
            Input array or scalar

        Returns
        -------
        np.ndarray
            Output array, same shape as input
        """
        if self.dropout is not None:
            # r will contain 0's and 1's, with approximately self.dropout % of 0's
            self.r = np.random.binomial(
                1.0,
                1 - self.dropout,
                input_array.shape,
            ).astype(np.float64)

            input_array *= self.r
        tmp = np.exp(input_array)
        self.Y = tmp / tmp.sum()
        return self.Y

    def back_grad(self, dLda: np.ndarray) -> None:
        r"""
        Sets derivative of loss w.r.t. input in grad dictionary

        Parameters
        ----------
        dLda : np.ndarray
            :math:`\frac{\partial L}{\partial Y}`:, must be
            same shape as input and output
        """
        g = dLda * self.Y
        self.grad["input"] = -np.sum(g) * self.Y + g
        if self.dropout is not None:
            self.grad["input"] *= self.r
