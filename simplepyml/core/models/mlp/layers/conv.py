from scipy.signal import correlate, convolve
from simplepyml.core.models.mlp.layers.layer import Layer
import numpy as np

# N-Dimensional Convolutional layer
# TODO: Padding?
class Conv(Layer):
    def __init__(
        self,
        activation,
        num_filters,
        filter_shape,
        dropout: float | np.floating = 0.0,
        *args,
        **kwargs,
    ):
        super().__init__()
        num_filters = int(num_filters)
        dropout = float(dropout)

        if num_filters < 1:
            raise TypeError("Number of filters for Conv layer must be > 0")
        self.activation_func = activation

        self.filter_shape = filter_shape
        self.num_filters = num_filters

        self.grad = dict()

        # TODO: Dropout
        self.dropout = dropout

    '''
    input_array shape: (# channels (RGB, etc.), x, y, z, w, ...)
    filter_shape: (x', y', z', w', ...). Dims = input_array.ndims - 1
    '''
    def _init_layer(self, input_array):
        self.initialized = True
        self.num_channels = input_array.shape[0]
        self.params["kernels"] = np.random.uniform(
            low=-1,
            high=1,
            size=(
                self.num_filters,
                self.num_channels,
            ) + self.filter_shape
        )
        self.params["biases"] = np.random.uniform(
            low=-1,
            high=-1,
            size=(self.num_filters,) + tuple(
                a - b + 1 for (a, b) in zip(
                    input_array.shape[1:],
                    self.filter_shape,
                )
            )
        )
        self.param_num = self.params["kernels"].size + self.params["biases"].size

    
    def __call__(self, input_array: np.ndarray):
        if not self.initialized:
            self._init_layer(input_array)
        self.input_array = input_array
        self.z = np.zeros(shape=self.params["biases"].shape)
        for i in range(self.num_filters):
            '''
            Equivalent Way of Calculating Correlation
            for j in range(self.num_channels):
                self.z[i] += correlate(
                    input_array[j],
                    self.params["kernels"][i][j],
                    mode="valid",
                    method="fft"
                )
            '''
            self.z[i] = correlate(
                input_array, 
                self.params["kernels"][i], 
                mode="valid",
                method="fft",
            )
        self.z += self.params["biases"]
        return self.activation_func(self.z)


    def back_grad(self, dLda: np.ndarray):
        phi_prime_z = self.activation_func.deriv(self.z)
        self.grad["biases"] = np.multiply(dLda, phi_prime_z)
        self.grad["kernels"] = np.zeros(shape=self.params["kernels"].shape)
        self.grad["input"] = np.zeros(shape=self.input_array.shape)
        for n in range(self.num_filters):
            for c in range(self.num_channels):
                self.grad["kernels"][n][c] = correlate(
                    self.input_array[c],
                    self.grad["biases"][n],
                    mode="valid",
                    method="fft",
                )
                self.grad["input"] += convolve(
                    dLda[n],
                    self.params["kernels"][n][c],
                    mode="full",
                    method="fft",
                )
        
        
