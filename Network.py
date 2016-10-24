import Layer
import OutputLayer

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