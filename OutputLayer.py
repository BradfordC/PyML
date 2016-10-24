import numpy as np
import math
import Functions

class OutputLayer:
    def __init__(self, size, errorFunction="MSE"):
        self.Size = size
        self.ErrorFunction = errorFunction

        self.Values = np.zeros(self.Size)
        self.Activations = np.zeros(self.Size)

    def Activate(self, values):
        if self.Values.size != values.size:
            print("Input size didn't match: {} vs {}".format(self.Values.size, values.size))
            return

        self.Values = values.copy()

        #Sigmoid
        for i in range(0, self.Values.size):
            self.Activations[i] = Functions.Sigmoid(self.Values[i])

    def GetOutput(self):
        return self.Activations.copy()

    def GetError(self, expected):
        if self.Values.size != expected.size:
            return

        if self.ErrorFunction == "MSE":
            errorSum = 0
            for i in range(0, self.Values.size):
                errorSum += math.pow(self.Values[i] - expected[i], 2)
            mse = errorSum / self.Values.size
            return mse

