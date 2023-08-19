import numpy as np
from simplepyml.util.functions.derivative import deriv

# All optimizing algorithms for training the MLP Neural Network

# Stoichastic Gradient Descent
# TODO: Implement + Batching Later
# TODO: Add loading bar for each epoch
def sgd(
    model, 
    input_data: np.ndarray,
    output_data: np.ndarray,
    epochs,
    learning_rate = 0.001,
):
    input_size = model.layers[0].size
    output_size = model.layers[-1].size
    for input in input_data:
        if len(input) != input_size:
            raise ValueError(
                f"Input shape mismatch: Expected {input_size}, got {len(input)}"
            )
    for output in output_data:
        if len(output) != output_size:
            raise ValueError(
                f"Input shape mismatch: Expected {output_size}, got {len(output)}"
            )
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        curr_loss = 0

        # dLdw = gradient of loss function with respect to weights (average of all dLdw_i's)
        dLdw = [
            np.zeros(shape = weight_layer.shape) for weight_layer in model.weights
        ]
        # dLdb = gradient of loss function with respect to biases (average of all dLdb_i's)
        dLdb = [
            np.zeros(shape = bias_layer.shape) for bias_layer in model.biases
        ]
        for input_data_index, input in enumerate(input_data):
            print(f"\rInput {input_data_index}/{len(input_data)}", end="\r")
            output = output_data[input_data_index]
            model.evaluate(input=input)
            curr_loss += model.loss(values=model.layers[-1].activation, expected = output)

            # Gradient of loss with respect to each activation,
            # used for calculating dLdw_i and dLdb_i
            dLda = [
                np.zeros(shape = layer.size) for layer in model.layers
            ]

            # Gradient of loss with respect to weights for this training example
            dLdw_i = [
                np.zeros(shape=weight_layer.shape) for weight_layer in model.weights
            ]

            # Gradient of loss with respect to biases for this training example
            dLdb_i = [
                np.zeros(shape = bias_layer.shape) for bias_layer in model.biases
            ]

            last_layer = True
            # Traverse through layers starting from the end

            # Custom dLda for last layer:

            dLda[-1] = deriv(model.loss)(model.layers[-1].activation, np.array(output))

            # Iterate backwards, excluding last layer since we calculated dLda for that already
            for layer_index in range(len(model.layers) - 2, -1, -1):
                # print(model.weights[layer_index])

                # Calculating dLda for this layer
                for i in range(len(dLda[layer_index])):
                    dLda[layer_index][i] = 0
                    for k in range(len(model.layers[layer_index + 1].activation)):
                        dLda[layer_index][i] += (
                            dLda[layer_index + 1][k] *
                            deriv(model.layers[layer_index + 1].activation_func)(
                                model.layers[layer_index + 1].z[k]
                            ) *
                            model.weights[layer_index][k][i]
                        )

                # Calculating dLdb_i for this layer
                for i in range(len(model.biases[layer_index])):
                    dLdb_i[layer_index][i] = (
                        dLda[layer_index + 1][i] *
                        deriv(model.layers[layer_index + 1].activation_func)(
                            model.layers[layer_index + 1].z[i]
                        )
                    )

                # Calculating dLdw_i for this layer
                for i in range(len(model.weights[layer_index])):                    
                    for j in range(len(model.weights[layer_index][i])):
                        dLdw_i[layer_index][i][j] = (
                            dLda[layer_index + 1][i] *
                            deriv(model.layers[layer_index + 1].activation_func)(
                                model.layers[layer_index + 1].z[i]
                            ) * 
                            model.layers[layer_index].activation[j]
                        )
            for i in range(len(dLdw)):
                dLdw[i] += dLdw_i[i]
            for i in range(len(dLdb)):
                dLdb[i] += dLdb_i[i]
        for w in dLdw:
            w /= len(input_data)
        for b in dLdb:
            b /= len(input_data)
        
        for i in range(len(model.weights)):
            model.weights[i] -= dLdw[i] * learning_rate
        for i in range(len(model.biases)):
            model.biases[i] -= dLdb[i] * learning_rate
        curr_loss /= len(input_data)
        print(f"Current Loss: {curr_loss}")