from simplepyml.core.models.mlp.layers.layer import Layer
from typing import Callable
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np

# Pooling layer
# TODO: Padding?


class Pooling(Layer):
    r"""
    N-Dimensional Pooling Layer.

    The pool_function is recommended to be a numpy aggregate function, such as np.mean, np.max, np.sum, np.std, etc.
    It must have an axis parameter which causes it to evaluate itself along specific numpy ndarray axes

    Parameters
    ----------
    pool_shape : tuple
        The shape of each pooling chunk. The length of pool_shape determines
        the dimension of this pooling layer.

        Ex: (4, 4) input and (3, 3) pool_shape will result in a (1, 1) output
        (one row and column is cut off since 4 % 3 == 1).

        Ex: (9, 9) input and (2, 2) pool_shape will result in a (4, 4) output
        (one row and column is cut off since 9 % 2 == 1)

        Ex: (2, 3) => 2D Pooling Layer, and the pool_function will be run on
        a 2d subset of the input matrix

        Ex: (3, 3, 1, 4) => 4D Pooling Layer, function will be run on 4d subset
        of input matrix
    pool_function : function {np.sum, np.mean, np.max; or cupy equivalents}
        Determines what kind of pooling layer this is (Max Pooling, Avg Pooling, Sum Pooling)

    Attributes
    ----------
    input_array : ndarray
        Most recent input of layer
    input_shape : tuple
        Shape of input array
    pool_shape : tuple
        pool shape, defined in constructor
    ndim : int
        Dimension of this pooling layer, equivalent to len(pool_shape)
    pool_function : function {np.sum, np.mean, np.max; or cupy equivalents}
        The pooling function
    pooling_size : int
        The product of every element in pool_shape. Used in backpropagation.
    params : dict()
        Empty dictionary; Pooling layer does not have trainable parameters
    grad : dict()
        Empty dictionary, until :py:func:`~back_grad` is called, after which
        grad["input"] is the loss gradient w.r.t. the input
    param_num : int (0)
        Number of parameters in this layer (0)
    initialized : bool
        Whether the layer has been initialized. False until called for the
        first time
    """

    def __init__(
        self,
        pool_shape,
        pool_function: Callable[[np.ndarray], np.number],
        *args,
        **kwargs,
    ):
        super().__init__()
        self.pool_shape = np.asarray(pool_shape)
        self.ndim = len(pool_shape)
        self.pool_function = pool_function
        self.grad = dict()

    def _init_layer(self, input_array):
        self.initialized = True
        self.input_shape = input_array.shape
        # Shape of axes where the pooling will actually occur
        self._pool_axes_shape = np.asarray(input_array.shape[-self.ndim :])
        # Shape of pooling layers (excluding higher dimensions)
        self._partitions = self._pool_axes_shape // self.pool_shape
        # Shape of input matrix with evenly divisible dimensions (cut off the excess)
        self._cut_matrix_shape = self._partitions * self.pool_shape

        # Used to cut off excess, using indexing. arr[this] will give a cut off matrix
        self._cut_matrix_index_tuple = (...,) + tuple(
            slice(int(i)) for i in self._cut_matrix_shape
        )

        # Index s.t. after higher non-pooling dimensions, you have p1, x1, p2, x2, ..., pn, xn
        # p_i is the i'th partition dimension; You can index along this to look at a certain chunk
        # x_i is the i'th index along that partition dimension
        self._indexed_cut_shape = self.input_shape[: -self.ndim] + tuple(
            int(i)
            for i in np.dstack((self._partitions, self.pool_shape)).reshape((-1,))
        )

        # Parameters for functions such as np.sum, np.mean, np.max, etc. Ex: np.sum(arr, axis=self._oveer_axes)
        self._over_axes = tuple(
            int(i)
            for i in np.arange(
                input_array.ndim - self.ndim + 1,
                2 * (self.ndim - 1) + input_array.ndim,
                2,
            )
        )

        self.output_shape = self.input_shape[: -self.ndim] + tuple(self._partitions)

        # Zoom factor along each dimension to upscale an output result to an input result
        # Ex: 5d input matrix with 3d pooling shape (1, 2, 3) will result in (1, 1, 1, 2, 3)
        self._zoom_factor = (1,) * (input_array.ndim - self.ndim) + tuple(
            int(i) for i in self.pool_shape
        )
        s_tuples = tuple(
            int(i)
            for i in [f * dim for f, dim in zip(self._zoom_factor, self.output_shape)]
        )

        # Used to index an output result in order to give an upscaled version with the same size as the cut off input result
        self._zoom_view = tuple(
            A // f
            for f, A in zip(
                self._zoom_factor, np.ogrid[tuple(slice(0, s) for s in s_tuples)]
            )
        )

        # Used to pad zoom result from the cutoff matrix shape to the actual input shape (used in np.pad)
        self._zoom_pad_values = tuple(
            zip(
                tuple(int(i) for i in np.zeros(shape=input_array.ndim, dtype="int32")),
                tuple(
                    int(i)
                    for i in np.concatenate(
                        (
                            np.zeros(shape=input_array.ndim - self.ndim, dtype="int32"),
                            (self._pool_axes_shape % self.pool_shape),
                        )
                    )
                ),
            )
        )
        self.pooling_size = self.pool_shape.prod()

    def __call__(self, input_array: np.ndarray):
        if not self.initialized:
            self._init_layer(input_array)
        self.input_array = input_array
        mat = input_array[self._cut_matrix_index_tuple].reshape(self._indexed_cut_shape)
        return self.pool_function(mat, axis=self._over_axes).astype(np.float64)

    def zoom(self, arr):
        """
        Zoom out an array

        Parameters
        ----------
        arr : ndarray
            The array to be zoomed out. arr.shape must be equal to the output shape.

        Returns
        -------
        ndarray
            Zoomed out array with shape equal to trimmed input shape

        """
        return np.pad(
            arr[self._zoom_view], self._zoom_pad_values, "constant", constant_values=0
        ).astype(np.float64)

    def back_grad(self, dLda: np.ndarray):
        """
        Backward gradient calculation. Given loss gradient w.r.t. the most
        recent output, calculate and store the loss gradient w.r.t. the most
        recent input, and store in grad["input"].

        Parameters
        ----------
        dLda : ndarray
            The loss gradient w.r.t. the most recent output. Must have shape
            equal to the output shape

        Returns
        -------
        None

        Raises
        ------
        NotImplementedException
            If the pool_function is not np.max, np.sum, np.avg, or cupy equivalents
        """
        self.grad["input"] = self.zoom(dLda)
        if self.pool_function == np.max:
            self.grad["input"][self.grad["input"] != self.input_array] = 0
        elif self.pool_function == np.mean:
            self.grad["input"] /= self.pooling_size
        elif self.pool_function == np.sum:
            return
        else:
            raise NotImplementedError("Gradient for this function is not implemented")
