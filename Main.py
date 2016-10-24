import GetInput
import Network
import numpy as np

#Data = GetInput.LoadAllCategories("D:/Chris/Documents/School/Machine Learning/Project/4Categories", True)

inputSize = 2
outputSize = 1

net = Network.Network(inputSize, outputSize, [3])

inputs = [[0,0],[0,1],[1,0],[1,1]]
outputs = [[0],[1],[1],[0]]

error = 10
a = 0
while error > .01:
    print(a, end="\t")
    for i in range(0, len(inputs)):
        input = np.array(inputs[i])
        expected = np.array(outputs[i])
        output = net.Feedforwad(input)
        error = net.GetError(expected)
        print(output[0], end="\t")
        net.Backprop(expected, .2)
    print(error)
    a+=1
