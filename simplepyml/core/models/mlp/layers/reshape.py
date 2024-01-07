from simplepyml.core.models.mlp.layers.layer import Layer
import numpy as np

# Reshape layer
class Reshape(Layer):
    def __init__(
        self,
        output_shape,
        *args,
        **kwargs,
    ):
        self.output_shape = output_shape
        self.grad = dict()
        self.params = dict()
        self.param_num = 0
        self.initialized = False

    def _init_layer(self, input_array):
        self.initialized = True
        self.input_shape = input_array.shape
    
    def __call__(self, input_array: np.ndarray):
        if not self.initialized:
            self._init_layer(input_array)
        return np.reshape(input_array, self.output_shape)

    def back_grad(self, dLda: np.ndarray):
        self.grad["input"] = np.reshape(dLda, self.input_shape)
