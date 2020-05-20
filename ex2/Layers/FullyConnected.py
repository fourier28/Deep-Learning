import numpy as np

class FullyConnected:
    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(0,1,(input_size+1 ,output_size))
        self.flag = 0
        self.input_tensor1 = None
        self.gradient_tensor = None

    def forward(self,input_tensor):

        bias = np.ones((input_tensor.shape[0], 1))
        input_tensor_bias = np.column_stack((input_tensor,bias))
        self.input_tensor1 = input_tensor_bias

        input_tensor_bias = np.dot(input_tensor_bias, self.weights)
        input_tensor = input_tensor_bias[:,0:self.output_size]

        return input_tensor


    def backward(self,error_tensor):
        #global gradient_tensor

        self.gradient_tensor = np.dot(self.input_tensor1.T, error_tensor)
        error_tensor = np.dot(error_tensor,self.weights.T)
        error_tensor = error_tensor[:,0:self.input_size]

        if self.flag == 1:
            self.weights = self.optimizer.calculate_update(self.weights,self.gradient_tensor)

        return error_tensor

    #def forward(self,input_tensor):

    #   self.input_tensor1 = input_tensor
    #  bias = np.ones((input_tensor.shape[0], 1))
    #    input_tensor_bias = np.column_stack((input_tensor,bias))


    #   input_tensor_bias = np.dot(input_tensor_bias, self.weights)
    #    input_tensor = input_tensor_bias[:,0:self.output_size]

     #   return input_tensor


    #def backward(self,error_tensor):
        #global gradient_tensor

    #    self.gradient_tensor = np.dot(self.input_tensor1.T, error_tensor)
    #    error_tensor = np.dot(error_tensor,self.weights.T)
    #    error_tensor = error_tensor[:,0:self.input_size]

    #    if self.flag == 1:
    #        self.weights[0:self.input_size,:] = self.optimizer.calculate_update(self.weights[0:self.input_size,:],self.gradient_tensor)

    #    return error_tensor

    @property
    def gradient_weights(self):
        return self.gradient_tensor

    @property
    def optimizer(self):
        return self._optimizer
    @optimizer.setter
    def optimizer(self,a):
        self._optimizer = a
        self.flag = 1



