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
    r"""
    Convolutional Layer.

    Definition:

    Takes an input, validly correlates it with a kernel matrix, adds a bias, and sends this result
    (z) through the activation function:

    .. math::
        z_{i,...} &= X \underset{valid}{\star} K_{i,...} + b_{i,...} \text{, for each filter } i

        Y &= \phi(z)

    Expected input array shape: (# Channels, x, y, z, w, ...)

    Filter shape: (x', y', z', w', ...) where each i' <= i

    Kernel shape: (# Filters, # Channels, x', y', z', w', ...)

    Output & Bias shape: (# Filters, x-x'+1, y-y'+1, z-z'+1, w-w'+1, ...) (due to valid correlation)

    The x, y, z, w, ... are the axes of correlation, and determines the dimension of this layer
    (1d convolutional layer vs. 2d, 3d, 4d, etc.)

    Note: Higher dimensions have not yet been tested.

    To calculate the loss gradient w.r.t. the biases, weights, and input, we use these formulas:

    .. math::
        \frac{\partial L}{\partial b} &= \frac{\partial L}{\partial Y} \odot \phi'(z)

        \frac{\partial L}{\partial K_{i,c,...}} &= X_{c,...} \underset{valid}{\star} \frac{\partial L}{\partial b_{i,...}}

        \frac{\partial L}{\partial X_{c,...}} &= \sum_{i=0}^{\text{num_filters - 1}} \frac{\partial L}{\partial b_{i,...}} \underset{full}{\ast} K_{i,c,...}

    My derivation of these formulas is here: :download:`pdf <../pdfs/convlayer_proof.pdf>`

    Parameters
    ----------
    activation : function
        Activation function. See :py:mod:`~simplepyml.util.functions.activation` functions
    num_filters : int
        Number of filters, length of axis 0 of output
    filter_shape : tuple
        Shape of the kernel filter. Determines the axes of correlation.
    dropout : float
        Currently useless, future plans to implement dropout.

    Attributes
    ----------
    input_array : ndarray
        Most recent input of layer
    z : ndarray
        Output of layer before put through activation function
    num_channels : int
        Number of filters, length of axis 0 of input and axis 1 of kernels array
    num_filters : int
        Number of filters, length of axis 0 of output
    dropout : float
        Currently useless.
    params : dict()
        Contains kernel and bias arrays
    grad : dict()
        Empty dictionary, until :py:func:`~back_grad` is called, after which:

        - grad["input"] is the loss gradient w.r.t. the input
        - grad["kernels"] is the loss gradient w.r.t. the kernels
        - grad["biases"] is the loss gradient w.r.t. the biases
    param_num : int
        Number of parameters in this layer (params["kernels"].size + params["biases"].size)
    initialized : bool
        Whether the layer has been initialized. False until called for the
        first time
    """

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
                a - b + 1  # Axis shape for valid correlation/convolution
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
        """
        Backward gradient calculation. Given loss gradient w.r.t. the most
        recent output, calculate and store the loss gradient w.r.t. the most
        recent input, kernels, and biases and stores in the grad dictionary.

        Parameters
        ----------
        dLda : ndarray
            The loss gradient w.r.t. the most recent output. Must have shape
            equal to the output shape

        Returns
        -------
        None
        """
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
        ).sum(axis=1)  # Add over all filters
