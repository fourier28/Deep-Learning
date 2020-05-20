import numpy as np

class CrossEntropyLoss:
    def __init__(self):
        self.input_tensor = None

    def forward(self,input_tensor,label_tensor):
        self.input_tensor = input_tensor
        Loss = 0

        #for i in range (label_tensor.shape[0]):
        #    for j in range (label_tensor.shape[1]):
        #        if label_tensor[i,j] == 1:
        #            Loss += -np.log(input_tensor[i,j]+np.finfo(float).eps)
        rows,cols = np.where(label_tensor == 1)
        for i in range(len(rows)):
                Loss += -np.log(input_tensor[rows[i],cols[i]] + np.finfo(float).eps)

        return Loss

    def backward(self,label_tensor):

        error_tensor = -label_tensor / self.input_tensor


        return error_tensor
