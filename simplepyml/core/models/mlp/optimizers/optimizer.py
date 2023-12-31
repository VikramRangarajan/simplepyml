from time import perf_counter

class Optimizer():
    def __init__(self):
        ...
    
    def __call__(self):
        ...
    
    def print_progress(index, num_inputs, accuracy, loss):
        print(
            f"Input {index+1}/{num_inputs};\t" +
            "Accuracy {:.3f};\t".format(accuracy/(index+1)) +
            "Loss {:.3f}".format(loss/(index+1)),
            end="\r",
        )
    
    def print_epoch(epoch_num, num_epochs, accuracy, loss, start_time):
        print(f"Epoch {epoch_num+1}/{num_epochs}")
        print()
        print("Current Loss: {:.3f}".format(loss))
        end_time = perf_counter()
        print("Current Accuracy: {:.3f}".format(accuracy))
        print("Time for epoch {}: {:.3f} seconds".format(epoch_num, end_time - start_time))
        return end_time - start_time