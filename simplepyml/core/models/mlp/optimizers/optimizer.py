from time import perf_counter


class Optimizer:
    """
    Base Optimizer Class

    """

    def __init__(self):
        ...

    def __call__(self):
        ...

    def print_progress(index, num_inputs, accuracy, loss):
        """

        Prints progress of the training loop for any subclass
        of optimizer

        Parameters
        ----------
        index : int
            Input index
        num_inputs : int
            Number of inputs
        accuracy : float
            Current accuracy
        loss : float
            Current loss
        """
        print(
            f"Input {index+1}/{num_inputs};\t"
            + "Accuracy {:.3f};\t".format(accuracy / (index + 1))
            + "Loss {:.3f}".format(loss / (index + 1)),
            end="\r",
        )

    def print_epoch(epoch_num, num_epochs, accuracy, loss, start_time):
        """
        Prints result of one epoch

        Parameters
        ----------
        epoch_num : int
            Epoch number (nth epoch)
        num_epochs : int
            Number of epochs to do
        accuracy : float
            Current accuracy
        loss : float
            Current loss
        start_time : float (time in seconds)
            Start time of epoch, used to calculate epoch runtime

        Returns
        -------
        float
            Runtime of epoch in seconds
        """
        print("\n")
        print(f"Epoch {epoch_num+1}/{num_epochs}")
        print()
        print("Current Loss: {:.3f}".format(loss))
        end_time = perf_counter()
        print("Current Accuracy: {:.3f}".format(accuracy))
        print(
            "Time for epoch {}: {:.3f} seconds".format(epoch_num, end_time - start_time)
        )
        return end_time - start_time
