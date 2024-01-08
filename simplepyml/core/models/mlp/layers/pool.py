from simplepyml.core.models.mlp.layers.layer import Layer
import numpy as np

# Pooling layer
# TODO: Padding?
'''
pool_shape is the shape of each chunk
The length of pool_shape determines the dimension of this pooling layer

Ex: (2, 3) => 2D Pooling Layer, and the pool_function will be run on a 2d subset of the input matrix
    (3, 3, 1, 4) => 4D Pooling Layer, function will be run on 4d subset of input matrix

The pool_function is recommended to be a numpy aggregate function, such as np.mean, np.max, np.sum, np.std, etc.
It must have an axis parameter which causes it to evaluate itself along specific numpy ndarray axes
'''
class Pooling(Layer):
    def __init__(
        self,
        pool_shape,
        pool_function,
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
        self._pool_axes_shape = np.asarray(input_array.shape[-self.ndim:])
        # Shape of pooling layers (excluding higher dimensions)
        self._partitions = self._pool_axes_shape // self.pool_shape
        # Shape of input matrix with evenly divisible dimensions (cut off the excess)
        self._cut_matrix_shape = self._partitions * self.pool_shape

        # Used to cut off excess, using indexing. arr[this] will give a cut off matrix
        self._cut_matrix_index_tuple = (...,) + tuple(slice(i) for i in self._cut_matrix_shape)

        # Index s.t. after higher non-pooling dimensions, you have p1, x1, p2, x2, ..., pn, xn
        # p_i is the i'th partition dimension; You can index along this to look at a certain chunk
        # x_i is the i'th index along that partition dimension
        self._indexed_cut_shape = self.input_shape[:-self.ndim] + tuple(
            np.dstack((self._partitions, self.pool_shape)).reshape((-1,))
        )

        # Parameters for functions such as np.sum, np.mean, np.max, etc. Ex: np.sum(arr, axis=self._oveer_axes)
        self._over_axes = tuple(
            np.arange(input_array.ndim - self.ndim + 1, 2 * (self.ndim - 1) + input_array.ndim, 2)
        )

        self.output_shape = self.input_shape[:-self.ndim] + tuple(self._partitions)

        # Zoom factor along each dimension to upscale an output result to an input result
        # Ex: 5d input matrix with 3d pooling shape (1, 2, 3) will result in (1, 1, 1, 2, 3)
        self._zoom_factor = (1,) * (input_array.ndim - self.ndim) + tuple(self.pool_shape)
        s_tuples = tuple(f * dim for f, dim in zip(self._zoom_factor, self.output_shape))

        # Used to index an output result in order to give an upscaled version with the same size as the cut off input result
        self._zoom_view = tuple(A//f for f, A in zip(self._zoom_factor, np.ogrid[tuple(slice(0, s) for s in s_tuples)]))

        # Used to pad zoom result from the cutoff matrix shape to the actual input shape (used in np.pad)
        self._zoom_pad_values = tuple(zip(
                tuple(np.zeros(shape=input_array.ndim, dtype="int32")),
                tuple(
                    np.concatenate(
                        (
                            np.zeros(shape=input_array.ndim - self.ndim, dtype="int32"), 
                            (self._pool_axes_shape % self.pool_shape)
                        )
                    )
                )
            )
        )
        self.pooling_size = self.pool_shape.prod()

    def __call__(self, input_array: np.ndarray):
        if not self.initialized:
            self._init_layer(input_array)
        self.input_array = input_array
        mat = input_array[self._cut_matrix_index_tuple].reshape(self._indexed_cut_shape)
        self.z = self.pool_function(mat, axis=self._over_axes)
        return self.z

    def zoom(self, arr):
        return np.pad(arr[self._zoom_view], self._zoom_pad_values, "constant", constant_values=0)

    def back_grad(self, dLda: np.ndarray):
        zoomed = self.zoom(dLda)
        if self.pool_function == np.max:
            zoomed[zoomed != self.input_array] = 0
            self.grad["input"] = zoomed
        elif self.pool_function == np.mean:
            self.grad["input"] = zoomed / self.pooling_size
        else:
            raise NotImplementedError("Gradient for this function is not implemented")
        
        
