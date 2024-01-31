from simplepyml.core.models.mlp.optimizers.optimizer import Optimizer
from simplepyml import USE_GPU
from tqdm import tqdm

if USE_GPU:
    import cupy as np
else:
    import numpy as np


class Adam(Optimizer):
    r"""
    The `Adam Optimizer`_ updates network parameters using the concept
    of velocity and momentum. It has proven to be a consistent and
    very fast (in terms of convergence) optimizer.

    .. _Adam Optimizer: https://doi.org/10.48550/arXiv.1412.6980

    """

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
        r"""
        Calling the Adam optimizer to run

        Parameters
        ----------
        model : :py:class:`simplepyml.core.models.mlp.mlp.MLP`
            The MLP model whose parameters will be updated
            using Adam
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
        learning_rate : float (default 0.001)
            The learning rate (stepsize / :math:`\alpha` ) in the Adam
            algorithm
        beta_1 : float (default 0.9)
            The exponential decay rate for first moment estimate
            (:math:`\beta_{1}`) in the Adam algorithm
        beta_2 : float (default 0.999)
            The exponential decay rate for second moment estimate
            (:math:`\beta_2`) in the Adam algorithm
        epsilon : float (default 1e-8)
            Small number, :math:`\epsilon > 0`

        Returns
        -------
        None
        """
        avg = 0

        for epoch in range(epochs):
            curr_loss, accuracy, tdiff = self.run_epoch(
                input_data, output_data, model, beta_1, beta_2, learning_rate, epsilon
            )
            curr_loss /= len(input_data)
            accuracy /= len(output_data)
            print(f"Epoch {epoch + 1} of {epochs} done, took ~{tdiff} seconds")
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
        r"""
        Runs one epoch

        Parameters
        ----------
        model : :py:class:`simplepyml.core.models.mlp.mlp.MLP`
            The MLP model whose parameters will be updated
            using Adam
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
        learning_rate : float (default 0.001)
            The learning rate (stepsize / :math:`\alpha` ) in the Adam
            algorithm
        beta_1 : float (default 0.9)
            The exponential decay rate for first moment estimate
            (:math:`\beta_{1}`) in the Adam algorithm
        beta_2 : float (default 0.999)
            The exponential decay rate for second moment estimate
            (:math:`\beta_2`) in the Adam algorithm
        epsilon : float (default 1e-8)
            Small number, :math:`\epsilon > 0`

        Returns
        -------
        tuple[float, float]
            Returns loss during the epoch, and the accuracy gained
            during that epoch (not entirely accurate)
        """
        curr_loss = 0
        accuracy = 0
        adam_t = 1
        bar = tqdm(range(len(input_data)), miniters=len(input_data) // 100, colour="green")
        for index in bar:
            output = output_data[index]
            output_result = model.evaluate(input=input_data[index])
            curr_loss += model.loss(values=output_result, expected=output)
            # Evaluates whether the evaluation is correct, to calculate accuracy
            # TODO: Implement accuracy function, maybe in loss objects
            if output_result.argmax() == output.argmax():
                accuracy += 1

            if (index) % int(len(input_data) / 100) == 0:
                bar.set_postfix({"accuracy": accuracy / (index + 1), "loss": curr_loss / (index + 1)}, refresh=True)

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
        return curr_loss, accuracy, bar.format_dict["elapsed"]
