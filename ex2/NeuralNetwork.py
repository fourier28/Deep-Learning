import copy
class NeuralNetwork:

    def __init__(self,a):
        self.optimizer = a
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.input =None

    def forward(self):

        self.input = self.data_layer.forward()
        output = self.layers[0].forward(self.input[0])
        for i in range(len(self.layers)-1):
            output = self.layers[i+1].forward(output)

        loss = self.loss_layer.forward(output, self.input[1])

        return loss

    def backward(self):

        #loss = self.loss_layer.forward(output, self.data_layer.forward()[1])
        e = self.loss_layer.backward(self.input[1])
        for i in range(len(self.layers)):
            e = self.layers[len(self.layers)-i-1].backward(e)

        return e

    def append_trainable_layer(self,layer):
        layer.optimizer = copy.deepcopy(self.optimizer)
        self.layers.append(layer)



    def train(self,iterations):
        for i in range(iterations):
            self.loss.append(self.forward())
            self.backward()

    def test(self,input_tensor):

        output_test = self.layers[0].forward(input_tensor)
        for i in range(len(self.layers) - 1):
            output_test = self.layers[i + 1].forward(output_test)

        return output_test


