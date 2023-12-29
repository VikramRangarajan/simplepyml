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

def run_epoch(
    input_data,
    output_data,
    model,
    learning_rate,
):
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
                f"Input {index+1}/{len(input_data)};\t" +
                "Accuracy {:.3f};\t".format(accuracy/(index+1)) +
                "Loss {:.3f}".format(curr_loss/(index+1)),
                end="\r",
            )
        
        dLda_next = (model.loss.deriv)(output_result, output)
        # Iterate backwards, excluding last layer since we calculated dLda for that already
        for layer_index in range(len(model.layers) - 1, -1, -1):
            layer = model.layers[layer_index]
            layer.back_grad(dLda_next)
            dLda_next = layer.dLdX
            layer.params["weights"] -= layer.dLdw * learning_rate
            layer.params["biases"] -= layer.dLdb * learning_rate
    return curr_loss, accuracy


def sgd(
    model, 
    input_data: np.ndarray,
    output_data: np.ndarray,
    epochs,
    learning_rate = 0.01,
):
    avg = 0
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        start_time = perf_counter()

        curr_loss, accuracy = run_epoch(input_data, output_data, model, learning_rate)

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