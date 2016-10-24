import Layer
import OutputLayer
import numpy as np

class Network:
    def __init__(self, inputSize, outputSize, hiddenLayerSizes, activationFunction="Sigmoid"):
        self.Layers = []

        layerSizes = [inputSize]
        layerSizes.extend(hiddenLayerSizes)
        layerSizes.append(outputSize)

        for i in range(0, len(layerSizes) - 1):
            newLayer = Layer.Layer(layerSizes[i], layerSizes[i+1], activationFunction)
            newLayer.RandomizeWeights()
            self.Layers.append(newLayer)
        self.Layers[0].Mode = "Linear"

        self.OutputLayer = OutputLayer.OutputLayer(outputSize)

    def Feedforwad(self, input):
        nextLayerInput = input.copy()
        for layer in self.Layers:
            nextLayerInput = layer.Feedforward(nextLayerInput)
        self.OutputLayer.Activate(nextLayerInput)
        return self.OutputLayer.GetOutput()

    def GetError(self, expected):
        return self.OutputLayer.GetError(expected)

    def Backprop(self, expected, learningRate):
        nextLayerDeltas = self.OutputLayer.GetDeltas(expected)
        for layer in reversed(self.Layers):
            gradient = np.dot(nextLayerDeltas.reshape(nextLayerDeltas.size, 1), layer.Activations.reshape(1,layer.Activations.size))
            layer.Weights = np.add(layer.Weights, np.multiply(gradient,learningRate))
            nextLayerDeltas = layer.MakeDeltas(nextLayerDeltas)