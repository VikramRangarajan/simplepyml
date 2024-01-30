class BaseLoss:
    """
    Base Loss Class.
    """

    def __init__(self):
        raise Exception("Do not use this class")

    def __call__(self, *args, **kwargs):
        ...

    def deriv(self, *args, **kwargs):
        ...


"""
TODO
def loss_function_from_str(x: str):
    x = x.lower().strip()
    if x == "mse":
        return mse.MSE()
    if x == "scce" or x == "sparsecategoricalcrossentropy":
        return scce.SCCE()
    
    raise ValueError("Invalid loss function")

def get_instance(loss_obj):
    if isinstance(loss_obj, BaseLoss):
        return loss_obj
    elif isinstance(loss_obj, str):
        return loss_function_from_str(loss_obj)
    else:
        new_obj = loss_obj()
        if isinstance(loss_obj, BaseLoss):
            return new_obj
        else:
            raise ValueError("Invalid Loss Function Object")
"""
