import GetInput
import Network
import numpy as np
import time
from random import randint
import math

#data = GetInput.LoadAllCategories("C:\\Users\\Chris\\Dropbox\\MLProjectResources\\4Categories", True)
data = GetInput.LoadAllCategories("D:\\Chris\\Documents\\School\\Machine Learning\\Project\\4Categories", True)
np.random.shuffle(data)

firstValidIndex = int(len(data) * .7)
firstTestIndex = int(len(data) * .85)

trainingSet = data[:firstValidIndex]
validationSet = data[firstValidIndex:firstTestIndex]
testingSet = data[firstTestIndex:]

print(len(trainingSet))

net = Network.Network(len(data[0][0]), len(data[0][1]), [255,255,255], 'ReLU')

groupSize = 250
groupCount = 0
testImageCount = 0
testImageCorrect = 0

testFile = open("C:\\Users\\Chris\\Dropbox\\School\\Machine Learning\\Project\\Checkpoint 5\\TestAccuracy.csv", 'w')
validFile = open("C:\\Users\\Chris\\Dropbox\\School\\Machine Learning\\Project\\Checkpoint 5\\ValidAccuracy.csv", 'w')

start = time.time()
for i in range(100):
    #Train one epoch on training set
    for image in trainingSet:
        input = np.array(image[0])
        expected = np.array(image[1])
        output = net.Feedforwad(input)
        error = net.GetError(expected)
        net.Backprop(expected, .0001)

        testImageCount += 1
        if np.argmax(expected) == randint(0,3):
            testImageCorrect += 1

        if testImageCount == groupSize:
            print(str(i) + " : " + str(groupCount))
            groupCount += 1
            groupAccuracy = testImageCorrect / groupSize
            testFile.write(str(time.time() - start) + ',' + str(groupAccuracy) + '\n')
            testImageCount = 0
            testImageCorrect = 0

    #Print validation accuracy after this epoch
    validImageCorrect = 0
    for image in validationSet:
        input = np.array(image[0])
        expected = np.array(image[1])
        output = net.Feedforwad(input)

        if np.argmax(expected) == np.argmax(output):
            validImageCorrect +=1
    validationAccuracy = validImageCorrect / len(validationSet)
    validFile.write(str(time.time() - start) + ',' + str(validationAccuracy) + '\n')

end = time.time()
print(str(end - start) + ' seconds.')