import numpy as np

class SoftMax:
    def __init__(self):
        pass
    def forward(self,input_tensor):
        global y
        E_input_tensor = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        input_tensor = E_input_tensor / np.sum(E_input_tensor, axis=1, keepdims=True)
        y = input_tensor
        return input_tensor

    def backward(self,error_tensor):

        error_tensor = y * (error_tensor - np.sum(error_tensor * y,axis=1,keepdims=True))

        return error_tensor


