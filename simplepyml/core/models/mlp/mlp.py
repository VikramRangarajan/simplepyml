# Multi Layer Perceptron Model
import numpy as np
from simplepyml.core.models.mlp.layers.layer import Layer

# MLP - Multi Level Perceptron Neural Network
class MLP():
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
        self.initialize_weights_and_biases()


    # Initializes the weight matrix and bias vector for each layer
    def initialize_weights_and_biases(self):
        self.weights = [
            np.zeros(
                shape=(self.layers[i+1].size, self.layers[i].size)
            ) for i in range(len(self.layers) - 1)
        ]


        self.biases = [
            np.zeros(
                shape=self.layers[i+1].size
            ) for i in range(len(self.layers) - 1)
        ]

    # Add layers to the MLP neural network
    def append_layers(self, layer: Layer | list[Layer]):
        if isinstance(layer, list):
            self.layers.extend(layer)
        else:
            self.layers.append(layer)
        self.initialize_weights_and_biases()
    
    def evaluate(self, input: np.ndarray | list):
        if len(input) != self.layers[0].size:
            raise ValueError(
                f"Invalid Input Size, received list of size {len(input)}, expected list of size {self.layers[0].size}"
            )
        
        # Sets the first layer's activations as the input
        self.layers[0].activation = np.array(input, dtype=np.float64)

        # Calculates the activations for each following layer
        for i, weights in enumerate(self.weights):
            self.layers[i+1].activation = self.layers[i].activation_func(np.matmul(weights, self.layers[i].activation) + self.biases[i])
        
        # Returns the result, which is the final layer's activations
        return self.layers[-1].activation