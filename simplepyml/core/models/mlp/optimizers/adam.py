from simplepyml.core.models.mlp.optimizers.optimizer import Optimizer
import numpy as np
from time import perf_counter

class Adam(Optimizer):
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
            tdiff = Adam.print_epoch(epoch, epochs, accuracy, curr_loss, start_time)
            avg += tdiff
        avg /= epochs
        print()
        print("Average time per epoch: {:.3f} seconds".format(avg))
    
    def run_epoch(
        self,
        input_data,
        output_data,
        model,
        beta_1=0.9,
        beta_2=0.999,
        learning_rate=0.001,
        epsilon=1e-8,
    ):
        curr_loss = 0
        accuracy = 0
        adam_t = 1
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
                Adam.print_progress(index, len(input_data), accuracy, curr_loss)
            
            dLda_next = (model.loss.deriv)(output_result, output)
            # Iterate backwards, excluding last layer since we calculated dLda for that already
            for layer_index in range(len(model.layers) - 1, -1, -1):
                layer = model.layers[layer_index]
                if not hasattr(layer, "_ADAM_INIT"):
                    layer._ADAM_INIT = True
                    for param in layer.params.keys():
                        setattr(layer, f"_adam_m_{param}", np.zeros(shape=layer.params[param].shape))
                        setattr(layer, f"_adam_v_{param}", np.zeros(shape=layer.params[param].shape))
                    # layer._adam_m = np.zeros(shape=layer.params["weights"].shape)
                    # layer._adam_m_bias = np.zeros(shape=layer.params["biases"].shape)
                    # layer._adam_v = np.zeros(shape=layer.params["weights"].shape)
                    # layer._adam_v_bias = np.zeros(shape=layer.params["biases"].shape)
                layer.back_grad(dLda_next)
                dLda_next = layer.grad["input"]
                # dLda_next = layer.dLdX

                for param in layer.params.keys():
                    setattr(
                        layer,
                        f"_adam_m_{param}",
                        beta_1 * getattr(layer, f"_adam_m_{param}") + (1 - beta_1) * layer.grad[param]
                    )
                    setattr(
                        layer,
                        f"_adam_v_{param}",
                        beta_2 * getattr(layer, f"_adam_v_{param}") + (1 - beta_2) * np.square(layer.grad[param])
                    )
                    m_hat = getattr(layer, f"_adam_m_{param}") / (1-(beta_1**adam_t))
                    v_hat = getattr(layer, f"_adam_v_{param}") / (1-(beta_2**adam_t))

                    layer.params[param] -= learning_rate * m_hat/(np.sqrt(v_hat) + epsilon)
                # layer._adam_m = beta_1 * layer._adam_m + (1 - beta_1) * layer.dLdw
                # layer._adam_m_bias = beta_1 * layer._adam_m_bias + (1 - beta_1) * layer.dLdb
                
                # layer._adam_v = beta_2*layer._adam_v + (1 - beta_2) * np.square(layer.dLdw)
                # layer._adam_v_bias = beta_2*layer._adam_v_bias + (1 - beta_2) * np.square(layer.dLdb)

                # m_hat = layer._adam_m / (1-(beta_1**adam_t))
                # m_hat_bias = layer._adam_m_bias / (1-beta_1**adam_t)
                
                # v_hat = layer._adam_v / (1-(beta_2**adam_t))
                # v_hat_bias = layer._adam_v_bias / (1-beta_2**adam_t)

                # layer.params["weights"] -= learning_rate*m_hat/(np.sqrt(v_hat) + epsilon)
                # layer.params["biases"] -= learning_rate*m_hat_bias/(np.sqrt(v_hat_bias) + epsilon)
            adam_t += 1
        for layer_index in range(len(model.layers) - 1, -1, -1):
            layer = model.layers[layer_index]
            del layer._ADAM_INIT
            for param in layer.params.keys():
                delattr(layer, f"_adam_m_{param}")
                delattr(layer, f"_adam_v_{param}")
        return curr_loss, accuracy