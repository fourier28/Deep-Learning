import numpy as np

class ReLU:
    def __init__(self):
        self.input_tensor = None
        pass
    def forward(self,input_tensor):

        self.input_tensor = input_tensor
        input_tensor = np.maximum(input_tensor,0)
        return input_tensor

    def backward(self,error_tensor):

        for i in range(self.input_tensor.shape[0]):
            for j in range(self.input_tensor.shape[1]):
                if self.input_tensor[i,j] <= 0:
                    error_tensor[i,j] = 0
                else:
                    pass

        #error_tensor = np.maximum(error_tensor,0)
        return error_tensor