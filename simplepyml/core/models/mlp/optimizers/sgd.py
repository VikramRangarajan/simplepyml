from simplepyml.core.models.mlp.optimizers.optimizer import Optimizer
import numpy as np
from time import perf_counter

class SGD(Optimizer):
    def __init__(self):
        ...

    def __call__(
        self,
        model, 
        input_data: np.ndarray,
        output_data: np.ndarray,
        epochs,
        learning_rate = 0.01,
    ):
        avg = 0
        
        for epoch in range(epochs):
            start_time = perf_counter()
            curr_loss, accuracy = self.run_epoch(input_data, output_data, model, learning_rate)
            curr_loss /= len(input_data)
            accuracy /= len(output_data)
            tdiff = SGD.print_epoch(epoch, epochs, accuracy, curr_loss, start_time)
            avg += tdiff
        avg /= epochs
        print()
        print("Average time per epoch: {:.3f} seconds".format(avg))
    
    def run_epoch(
        self,
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
            if len(input_data) > 5000 and (index+1) % (len(input_data) // 100) == 0:
                SGD.print_progress(index, len(input_data), accuracy, curr_loss)
            
            dLda_next = (model.loss.deriv)(output_result, output)
            # Iterate backwards, excluding last layer since we calculated dLda for that already
            for layer_index in range(len(model.layers) - 1, -1, -1):
                layer = model.layers[layer_index]
                layer.back_grad(dLda_next)
                dLda_next = layer.grad["input"]
                for param in layer.params.keys():
                    layer.params[param] -= layer.grad[param] * learning_rate
        return curr_loss, accuracy