import GetInput
import Network
import numpy as np
import random

def GetAccuracy(set, network):
    correctCount = 0

    for image in set:
        input = np.array(image[0])
        expected = np.array(image[1])
        output = net.Feedforwad(input)

        if np.argmax(expected) == np.argmax(output):
            correctCount += 1

    return correctCount / len(set)


# data = GetInput.LoadAllCategories("D:/Chris/Documents/School/Machine Learning/Project/4Categories", True)
data = GetInput.LoadAllCategories("D:/Chris/Documents/School/Machine Learning/Project/SmallSet", True)
random.shuffle(data)

firstValidIndex = int(len(data) * .7)
firstTestIndex = int(len(data) * .85)

trainingSet = data[:firstValidIndex]
validationSet = data[firstValidIndex:firstTestIndex]
testingSet = data[firstTestIndex:]

net = Network.Network(len(data[0][0]), len(data[0][1]), [200])

groupCount = 0
groupSize = 250

count = 0
correct = 0

maxAcc = 0

for i in range(5):
    for image in trainingSet:
        input = np.array(image[0])
        expected = np.array(image[1])
        output = net.Feedforwad(input)
        error = net.GetError(expected)
        net.Backprop(expected, .01)

        category = np.argmax(expected)
        classify = np.argmax(output)

        if category == classify:
            correct += 1
        count += 1

        if count == groupSize:
            print(groupCount, end= "\t")
            print(correct / count, end='\t')

            validAccuracy = GetAccuracy(validationSet, net)
            print(validAccuracy)
            if validAccuracy > maxAcc:
                maxAcc = validAccuracy

            groupCount += 1
            correct = 0
            count = 0

print(maxAcc)


