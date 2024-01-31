# Multi Layer Perceptron Model
from simplepyml.core.models.mlp.layers.layer import Layer
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# MLP - Multi Level Perceptron Neural Network
class MLP:
    """
    Multi-Layer Perceptron (MLP) Model

    Parameters
    ----------
    layers : :py:class:`.Layer` | list[:py:class:`.Layer`]
        List of layers in the MLP. Can be appended to.
    optimizer : :py:class:`.Optimizer`
        Optimizer object required
    loss_function : :py:class:`.BaseLoss`
        Loss function object
    """

    def __init__(
        self,
        layers: Layer | list[Layer],
        optimizer,
        loss_function,
    ):
        self.optimizer = optimizer
        self.loss = loss_function

        if isinstance(layers, Layer):
            self.layers = [layers]
        else:
            self.layers = layers

    # Add layers to the MLP neural network
    def append_layers(self, layer: Layer | list[Layer]):
        """
        Adds a layer to the MLP.

        Warnings
        --------
        DO NOT USE WHILE TRAINING!

        Parameters
        ----------
        layer : Layer | list[Layer]
            Layer or layers to add
        """
        if isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    def evaluate(self, input: np.ndarray):
        """
        Evaluates an input by putting it through the MLP.

        Parameters
        ----------
        input : ndarray
            Input array

        Returns
        -------
        ndarray
            Output array
        """
        inp = input
        for i in range(len(self.layers)):
            inp = self.layers[i](inp)
        return inp

    # TODO: Implement Batching
    def train(
        self, input_data: np.ndarray, output_data: np.ndarray, epochs, *args, **kwargs
    ):
        """
        Train the model given training data.

        Parameters
        ----------
        input_data : ndarray
            This input data has shape (# inputs, ...). Axis
            0 of this array will be iterated over to get every
            single input to the network. **This should be training
            data**.
        output_data : ndarray
            This output data has shape (# inputs, ...) where ... is
            related to the output shape of the network. A dynamic way
            of if an input results in a correct output is a TODO.
        epochs : int
            The number of epochs to run
        """
        self.optimizer(
            self,
            input_data=input_data,
            output_data=output_data,
            epochs=epochs,
            *args,
            **kwargs,
        )
