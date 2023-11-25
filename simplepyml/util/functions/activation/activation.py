class BaseActivation:
    def __init__(self):
        raise Exception("Do not use this class")
    
    def __call__(self, *args, **kwargs):
        ...
    
    def deriv(self, *args, **kwargs):
        ...
'''
TODO
def activation_function_from_str(x: str):
    x = x.lower().strip()
    if x == "sigmoid":
        return sigmoid.Sigmoid()
    if x == "relu":
        return relu.Relu()
    if x == "linear":
        return linear.Linear()
    
    raise ValueError("Invalid activation function")

def get_instance(loss_obj):
    if isinstance(loss_obj, BaseActivation):
        return loss_obj
    else:
        new_obj = loss_obj()
        if isinstance(loss_obj, BaseActivation):
            return new_obj
        else:
            raise ValueError("Invalid Activation Function Object")
'''