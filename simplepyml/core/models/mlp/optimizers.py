import numpy as np
from simplepyml.util.functions.derivative import deriv
from time import perf_counter

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
                f"Output shape mismatch: Expected {output_size}, got {len(output)}"
            )
    
    for epoch in range(epochs):
        print(f"Epoch {epoch}")
        start_time = perf_counter()
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
            # This print is very inefficient
            # if input_data_index % 100 == 0:
            #     print(f"\rInput {input_data_index}/{len(input_data)}", end="\r")
            output = output_data[input_data_index]
            output_result = model.evaluate(input=input)
            curr_loss += model.loss(values=output_result, expected = output)

            # Gradient of loss with respect to each activation,
            # used for calculating dLdw_i and dLdb_i
            dLda = [
                None for _ in model.layers
            ]

            # Gradient of loss with respect to weights for this training example
            dLdw_i = [
                np.zeros(shape=weight_layer.shape) for weight_layer in model.weights
            ]

            # Gradient of loss with respect to biases for this training example
            dLdb_i = [
                None for _ in model.biases
            ]
            # Traverse through layers starting from the end

            # Custom dLda for last layer:
            dLda[-1] = deriv(model.loss)(output_result, output)
            # Iterate backwards, excluding last layer since we calculated dLda for that already
            for layer_index in range(len(model.layers) - 2, -1, -1):
                # Calculating dLda for this layer
                phi_prime_z = deriv(model.layers[layer_index + 1].activation_func)(model.layers[layer_index + 1].z)
                new_matrix = dLda[layer_index + 1] * phi_prime_z
                dLda[layer_index] = np.matmul(model.weights[layer_index].T, new_matrix)

                # Calculating dLdb_i for this layer
                dLdb_i[layer_index] = new_matrix
                
                # Calculating dLdw_i for this layer
                # Numpy differentiates between (n,) and (n, 1) arrays, unfortunately.
                # np.atleast_2d = np.reshape(-1, 1)
                dLdw_i[layer_index] = np.matmul(
                    np.atleast_2d(new_matrix).T, 
                    np.atleast_2d(model.layers[layer_index].activation),
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
        end_time = perf_counter()
        print(f"Time for epoch {epoch}: {end_time - start_time}")