from simplepyml.core.models.mlp.layers.layer import Layer
import numpy as np

# Dense layer
class Dense(Layer):
    def __init__(
        self,
        size: int | np.integer,
        activation,
        dropout: float | np.floating = 0.0,
        *args,
        **kwargs,
    ):
        size = int(size)
        # TODO: dropout
        dropout = float(dropout)
        
        if size < 1:
            raise ValueError("Size of dense layer must be >= 1")

        self.activation_func = activation

        self.size = size

        self.grad = dict()

        self.dropout = dropout
        self.initialized = False

    def _init_layer(self, input_array):
        self.initialized = True
        input_size = input_array.shape[-1]
        self.params = {
            "weights": np.random.uniform(size=(self.size, input_size), low=-1, high=1),
            "biases": np.random.uniform(size=self.size, low=-1, high=1)
        }
        self.param_num = self.params["weights"].size + self.params["biases"].size

    def __call__(self, input_array: np.ndarray):
        if not self.initialized:
            self._init_layer(input_array)
        self.input_array = input_array
        self.z = self.params["weights"]@input_array + self.params["biases"]
        return self.activation_func(self.z)

    def back_grad(self, dLda: np.ndarray):
        phi_prime_z = self.activation_func.deriv(self.z)
        self.grad["biases"] = np.multiply(dLda, phi_prime_z)
        self.grad["input"] = self.params["weights"].T@self.grad["biases"]
        self.grad["weights"] = np.reshape(self.grad["biases"], (1, -1)).T@np.reshape(self.input_array, (1, -1))
