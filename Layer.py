import numpy as np
import Functions

class Layer:
    def __init__(self, size, nextLayerSize, activationMode="Sigmoid"):
        self.Size = size
        self.NodeCount = size + 1

        self.Values = np.zeros(self.Size)
        self.Activations = np.zeros(self.NodeCount)
        self.Weights = np.zeros(shape=(self.NodeCount, nextLayerSize))

        self.Mode = activationMode

    def Feedforward(self, values):
        if(self.Values.size != values.size):
            print("Input size didn't match: {} vs {}".format(self.Values.size, values.size))
            return

        self.Values = values.copy()
        self.__Activate()
        Output = np.dot(self.Activations, self.Weights)
        return Output

    def RandomizeWeights(self):
        for row in range(0, self.Weights.shape[0]):
            for column in range(0, self.Weights.shape[1]):
                self.Weights[row,column] = np.random.normal(0, 1)

    def __Activate(self):
        if self.Mode == "Linear":
            for i in range(0, self.Values.size):
                self.Activations[i] = self.Values[i]

        if self.Mode == "Sigmoid":
            for i in range(0, self.Values.size):
                self.Activations[i] = Functions.Sigmoid(self.Values[i])

        if self.Mode == "ReLU":
            for i in range(0, self.Values.size):
                self.Activations[i] = max(self.Values[i], 0)

        # Bias node
        self.Activations[-1] = 1




