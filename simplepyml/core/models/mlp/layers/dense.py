# Dense layer
from layer import Layer
import numpy as np
import simplepyml.util.functions.activation as af


class Dense(Layer):
    def __init__(
        self,
        size: int | np.integer,
        activation: function | str,
        dropout: float | np.floating = 0.0,
    ):
        if isinstance(activation, str):
            self.activation = af.activation_function_from_str(activation)
        else:
            self.activation = activation
