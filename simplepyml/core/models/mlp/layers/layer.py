# Layer superclass


class Layer:
    r"""
    Base Layer.

    ...

    Attributes
    ----------
    params : dict()
        Empty dictionary; Base layer does not have trainable parameters
    param_num : int (0)
        Number of parameters in this layer (0)
    initialized : bool (False)
        Whether the layer has been initialized. Always False.
    """

    def __init__(self, *args, **kwargs):
        self.params = dict()
        self.param_num = 0
        self.initialized = False

    def __call__():
        ...

    def back_grad():
        ...
