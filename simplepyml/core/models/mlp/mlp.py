# Multi Layer Perceptron Model
from simplepyml.core.models.mlp.layers.layer import Layer
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


# MLP - Multi Level Perceptron Neural Network
class MLP:
    def __init__(
        self,
        layers: None | Layer | list[Layer],
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
        if isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)

    def evaluate(self, input: np.ndarray | list):
        inp = self.layers[0](input)
        for i in range(1, len(self.layers)):
            inp = self.layers[i](inp)
        return inp

    # TODO: Implement Batching
    def train(self, input_data: np.ndarray, output_data, epochs, *args, **kwargs):
        self.optimizer(
            self,
            input_data=input_data,
            output_data=output_data,
            epochs=epochs,
            *args,
            **kwargs,
        )
