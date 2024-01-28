from simplepyml.core.models.mlp.layers.layer import Layer
from typing import Callable
from simplepyml import USE_GPU

if USE_GPU is True:
    import cupy as np
    from cupyx.scipy.signal import fftconvolve
else:
    import numpy as np
    from scipy.signal import fftconvolve


# N-Dimensional Convolutional layer
# TODO: Padding?
class Conv(Layer):
    def __init__(
        self,
        activation: Callable[[np.ndarray], np.ndarray],
        num_filters: int | np.integer,
        filter_shape: tuple,
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

    """
    input_array shape: (# channels (RGB, etc.), x, y, z, w, ...)
    filter_shape: (x', y', z', w', ...). Dims = input_array.ndims - 1
    """

    def _init_layer(self, input_array: np.ndarray) -> None:
        self.initialized = True
        self.num_channels = input_array.shape[0]
        self.params["kernels"] = np.random.uniform(
            low=-1,
            high=1,
            size=(
                self.num_filters,
                self.num_channels,
            )
            + self.filter_shape,
        )
        self.params["biases"] = np.random.uniform(
            low=-1,
            high=-1,
            size=(self.num_filters,)
            + tuple(
                a - b + 1 # Axis shape for valid correlation/convolution
                for (a, b) in zip(
                    input_array.shape[1:],
                    self.filter_shape,
                )
            ),
        )
        self.param_num = self.params["kernels"].size + self.params["biases"].size

        # Axes to convolve over, with relation to input array with new axis to match up 
        # with kernel shape (all except axis 0)
        self._forward_axes = tuple(range(1, input_array.ndim + 1))

        # Axes to convolve (all excluding axis 0, 1 which are filters & channels)
        self._backward_kernel_axes = tuple(range(2, input_array.ndim + 1))

        # Axes to flip over in order to do a correlation using fftconvolve(), ignoring axis 0 (filters)
        self._conv_to_corr_flip_axes = tuple(range(1, self.params["biases"].ndim))

        # Axes to convolve over to calculate input gradient (excludes the num_filters and channels axes)
        self._backward_input_axes = tuple(range(2, self.params["kernels"].ndim))

    def __call__(self, input_array: np.ndarray) -> np.ndarray:
        if not self.initialized:
            self._init_layer(input_array)
        self.input_array = input_array
        # Increase input dimension by 1 to match kernel shape, so correlation can be broadcasted over axis 0
        self.z = (
            fftconvolve(
                input_array[None],
                # Using convolve function, so must flip and conjugate in2 to get a correlation
                np.flip(self.params["kernels"]).conj(),
                "valid",
                self._forward_axes,
            )[:, 0, ...]
            + self.params["biases"]
        )
        # Referring to the [:, 0, ...] indexing:
        # As num_channels (axis 1 of input_array[None] and kernels) are equal, the valid convolution will
        # result in a size of 1 on axis 1, so we must reduce it by 1 dimension to match the output shape
        return self.activation_func(self.z)

    def back_grad(self, dLda: np.ndarray) -> None:
        phi_prime_z = self.activation_func.deriv(self.z)
        self.grad["biases"] = np.multiply(dLda, phi_prime_z)

        """
        The kernel gradient calculation requires a double nested for loop, looping from 
        [0, num_filters] and [0, num_channnels]
        It performed correlations in an order that would follow a cartesian product:
        (0, 0), (0, 1), ..., (0, m), (1, 0), (1, 1), ..., (1, m), ..., (n, 0), (n, 1), ..., (n, m)
        For loops are avoided by using broadcasting on axis 0 of self.input_array[None] and axis 1 of:
        np.flip(self.grad["biases"], axis=self._conv_to_corr_flip_axes).conj()[:, None, ...]

        We must flip and conjugate the second input because we are using fftconvolve() and we
        need a correlation for this calculation (mimicking scipy behavior)
        """
        self.grad["kernels"] = fftconvolve(
            # Expand axis 0 (1 "filter")
            self.input_array[None],
            # Expand axis 1 (1 "channel")
            np.flip(
                self.grad["biases"],
                axis=self._conv_to_corr_flip_axes,
            ).conj()[:, None, ...],
            mode="valid",
            axes=self._backward_kernel_axes,
        )
        self.grad["input"] = fftconvolve(
            # Expand axis 0 (channels, filters, ...)
            self.grad["biases"][None],
            # Adjusted index to match with above: (filters, channels, ... -> channels, filters, ...)
            self.params["kernels"].swapaxes(0, 1),
            mode="full",
            axes=self._backward_input_axes,
        ).sum(axis=1) # Add over all filters
