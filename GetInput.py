from PIL import Image
import os
import colorsys

def GetInput(image):
    imageData = image.load()

    inputArray = []
    for y in range(0, image.height):
        for x in range(0, image.width):
            pixel = imageData[x,y]
            inputArray.append(pixel[0])#R
            inputArray.append(pixel[1])#G
            inputArray.append(pixel[2])#B

    inputArray = Normalize(inputArray)
    return inputArray

def GetLocalizedInput(image):
    imageData = image.load()

    inputArray = []
    #Get average values per row
    for row in range(0, image.height):
        rSum = 0.0
        gSum = 0.0
        bSum = 0.0

        for x in range(0, image.width):
            pixel = imageData[x, row]
            rSum += pixel[0]
            gSum += pixel[1]
            bSum += pixel[2]
        averageR = rSum / image.width
        averageG = gSum / image.width
        averageB = bSum / image.width

        inputArray.append(averageR)
        inputArray.append(averageG)
        inputArray.append(averageB)

    #Get average values per column
    for column in range(0, image.width):
        rSum = 0.0
        gSum = 0.0
        bSum = 0.0

        for y in range(0, image.height):
            pixel = imageData[column, y]
            rSum += pixel[0]
            gSum += pixel[1]
            bSum += pixel[2]
        averageR = rSum / image.width
        averageG = gSum / image.width
        averageB = bSum / image.width

        inputArray.append(averageR)
        inputArray.append(averageG)
        inputArray.append(averageB)

    inputArray = Normalize(inputArray)
    return inputArray

def GetHistograms(image):
    imageData = image.load()

    firstHistogram = image.histogram()

    inputArray = []
    for y in range(0, image.height):
        for x in range(0, image.width):
            pixel = imageData[x,y]
            hsv = colorsys.rgb_to_hsv(pixel[0],pixel[1],pixel[2])
            hsv = (int(hsv[0]),int(hsv[1]),int(hsv[2]))

            image.putpixel((x,y), hsv)

    histogram = image.histogram()
    smallHistogram = []

    reductionFactor = 8

    for i in range(int(len(histogram) / reductionFactor)):
        sum = 0
        for j in range(reductionFactor):
            sum += histogram[i * reductionFactor + j]
        smallHistogram.append(sum)


    return firstHistogram


def LoadCategory(categoryFolderPath, mode):
    inputs = []
    for imageName in sorted(os.listdir(categoryFolderPath)):
        imagePath = os.path.join(categoryFolderPath, imageName)
        #If the path is actually another folder, search it for images as well
        if(os.path.isdir(imagePath)):
            print(imagePath)
            inputs.extend(LoadCategory(imagePath, mode))
            continue
        #Try to load an image
        try:
            image = Image.open(imagePath)
        except IOError:
            #Ignore it if it's not an image
            continue
        image = ScaleTo(image, 20, 20)
        if(mode == 'Full'):
            inputs.append(GetInput(image))
        elif(mode == 'Fast'):
            inputs.append(GetLocalizedInput(image))
        elif(mode == 'Hist'):
            inputs.append(GetHistograms(image))
        else:
            print('Error')
    return inputs

def LoadAllCategories(folderPath, mode):
    #Each data point is a pair of input and output
    data = []

    #Get all subfolders in folder
    categoryFolders = sorted(os.listdir(folderPath))
    categoryFolders = [os.path.join(folderPath, category) for category in categoryFolders]
    categoryFolders = [x for x in categoryFolders if os.path.isdir(x)]

    numCategories = len(categoryFolders)
    category = 0
    for categoryPath in categoryFolders:
        inputs = LoadCategory(categoryPath, mode)
        output = [0] * numCategories
        output[category] = 1

        for input in inputs:
            # Clone output here to avoid reference issues
            data.append([input, output[:]])

        category += 1
        
    return data

def Normalize(input):
    return [x/255 for x in input]

def ScaleTo(image, width, height):
    return image.resize((width, height), Image.ANTIALIAS)