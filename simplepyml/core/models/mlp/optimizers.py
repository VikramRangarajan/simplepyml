import numpy as np
# from simplepyml.util.functions.derivative import deriv
from time import perf_counter

# All optimizing algorithms for training the MLP Neural Network

# Stoichastic Gradient Descent
# TODO: Implement Batching
'''
Current Algorithm:
- Iterate over epochs
- Iterate over input data
    - Calculate dL/da for last layer
    - Iterate over layers, backwards, excluding last layer
    - Calculate weight & bias gradients, add it to model's weights (*learning_rate)

'''
def sgd(
    model, 
    input_data: np.ndarray,
    output_data: np.ndarray,
    epochs,
    learning_rate = 0.01,
):
    avg = 0
    input_size = model.layers[0].size
    output_size = model.layers[-1].size
    if input_data.shape[1] != input_size:
        raise ValueError(
                f"Input shape mismatch: Expected {input_size}, got {input_data.shape[1]}"
            )
    if output_data.shape[1] != output_size:
        raise ValueError(
            f"Output shape mismatch: Expected {output_size}, got {output_data.shape[1]}"
        )
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = perf_counter()
        curr_loss = 0
        accuracy = 0
        for index in range(len(input_data)):
            output = output_data[index]
            output_result = model.evaluate(input=input_data[index])
            curr_loss += model.loss(values=output_result, expected = output)

            # Evaluates whether the evaluation is correct, to calculate accuracy
            # TODO: Implement accuracy function, maybe in loss objects
            if output_result.argmax() == output.argmax():
                accuracy += 1
            
            # This print is very inefficient
            if (index+1) % (len(input_data) // 100) == 0 and len(input_data) > 5000:
                print(
                    f"Input {index+1}/{len(input_data)}; \
                        Accuracy {accuracy/(index+1)}; \
                            Loss {curr_loss/(index+1)}",
                    end="\r",
                )
            
            dLda_next = (model.loss.deriv)(output_result, output)
            # Iterate backwards, excluding last layer since we calculated dLda for that already
            for layer_index in range(len(model.layers) - 2, -1, -1):
                # Calculating dLda for this layer
                phi_prime_z = (model.layers[layer_index + 1].activation_func.deriv)(model.layers[layer_index + 1].z)
                new_matrix = np.multiply(dLda_next, phi_prime_z)
                dLda_next = np.matmul(model.weights[layer_index].T, new_matrix)

                # Calculating dLdb_i for this layer
                dLdb_i = new_matrix
                
                # Calculating dLdw_i for this layer
                # Numpy differentiates between (n,) and (n, 1) arrays, unfortunately.
                # np.atleast_2d = np.reshape(1, -1)
                dLdw_i = np.matmul(
                    np.reshape(new_matrix, (1, -1)).T, 
                    np.reshape(model.layers[layer_index].activation, (1, -1)),
                )
                model.weights[layer_index] -= dLdw_i * learning_rate
                model.biases[layer_index] -= dLdb_i * learning_rate

        curr_loss /= len(input_data)
        accuracy /= len(output_data)
        print()
        print(f"Current Loss: {curr_loss}")
        end_time = perf_counter()
        print(f"Current Accuracy: {accuracy}")
        print(f"Time for epoch {epoch}: {end_time - start_time}")
        avg += end_time - start_time
    avg /= epochs
    print(avg)