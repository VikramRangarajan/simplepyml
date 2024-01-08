# Layer superclass

class Layer():
    def __init__(self, *args, **kwargs):
        self.params = dict()
        self.param_num = 0
        self.initialized = False
    
    def __call__():
        ...
    
    def back_grad():
        ...