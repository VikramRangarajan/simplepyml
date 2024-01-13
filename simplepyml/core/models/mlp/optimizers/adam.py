from simplepyml.core.models.mlp.optimizers.optimizer import Optimizer
from time import perf_counter
from simplepyml import USE_GPU

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Adam(Optimizer):
    def __init__(self):
        ...

    def __call__(
        self,
        model,
        input_data: np.ndarray,
        output_data: np.ndarray,
        epochs: int | np.integer,
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
    ) -> None:
        avg = 0

        for epoch in range(epochs):
            start_time = perf_counter()
            curr_loss, accuracy = self.run_epoch(
                input_data, output_data, model, beta_1, beta_2, learning_rate, epsilon
            )
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
    ) -> tuple[float, float]:
        curr_loss = 0
        accuracy = 0
        adam_t = 1
        for index in range(len(input_data)):
            output = output_data[index]
            output_result = model.evaluate(input=input_data[index])
            curr_loss += model.loss(values=output_result, expected=output)

            # Evaluates whether the evaluation is correct, to calculate accuracy
            # TODO: Implement accuracy function, maybe in loss objects
            if output_result.argmax() == output.argmax():
                accuracy += 1

            # This print is very inefficient
            if len(input_data) > 5000 and (index + 1) % (len(input_data) // 100) == 0:
                Adam.print_progress(index, len(input_data), accuracy, curr_loss)

            dLda_next = (model.loss.deriv)(output_result, output)
            # Iterate backwards, excluding last layer since we calculated dLda for that already
            for layer_index in range(len(model.layers) - 1, -1, -1):
                layer = model.layers[layer_index]
                if not hasattr(layer, "_ADAM_INIT"):
                    layer._ADAM_INIT = True
                    for param in layer.params.keys():
                        setattr(
                            layer,
                            f"_adam_m_{param}",
                            np.zeros(shape=layer.params[param].shape),
                        )
                        setattr(
                            layer,
                            f"_adam_v_{param}",
                            np.zeros(shape=layer.params[param].shape),
                        )
                layer.back_grad(dLda_next)
                dLda_next = layer.grad["input"]

                for param in layer.params.keys():
                    setattr(
                        layer,
                        f"_adam_m_{param}",
                        beta_1 * getattr(layer, f"_adam_m_{param}")
                        + (1 - beta_1) * layer.grad[param],
                    )
                    setattr(
                        layer,
                        f"_adam_v_{param}",
                        beta_2 * getattr(layer, f"_adam_v_{param}")
                        + (1 - beta_2) * np.square(layer.grad[param]),
                    )
                    m_hat = getattr(layer, f"_adam_m_{param}") / (1 - (beta_1**adam_t))
                    v_hat = getattr(layer, f"_adam_v_{param}") / (1 - (beta_2**adam_t))

                    layer.params[param] -= (
                        learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
                    )
            adam_t += 1
        for layer in model.layers:
            del layer._ADAM_INIT
            for param in layer.params.keys():
                delattr(layer, f"_adam_m_{param}")
                delattr(layer, f"_adam_v_{param}")
        return curr_loss, accuracy
