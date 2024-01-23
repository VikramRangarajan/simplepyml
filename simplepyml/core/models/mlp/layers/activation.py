from simplepyml.core.models.mlp.layers.layer import Layer
from typing import Callable
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# Reshape layer
class Reshape(Layer):
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

    def back_grad(self, dLda: np.ndarray):
        self.grad["input"] = dLda * self.activation(self.input_array)

