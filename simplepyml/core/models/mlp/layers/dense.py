from simplepyml.core.models.mlp.layers.layer import Layer
import numpy as np

# Dense layer


class Dense(Layer):
    def __init__(
        self,
        size: int | np.integer,
        activation: str,
        dropout: float | np.floating = 0.0,
    ):
        if not isinstance(size, np.integer) and type(size) != int:
            raise TypeError("Size of dense layer must be an integer")
        
        if size < 1:
            raise ValueError("Size of dense layer must be >= 1")

        self.activation_func = activation

        self.activation = np.zeros(shape=size, dtype=np.float64)

        # Activation values before put through activation function,
        # used for calculating gradient in SGD
        self.z = np.zeros(shape=size, dtype=np.float64)
        self.size = size

        self.dropout = dropout
