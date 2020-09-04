import cv2
import numpy as np
from typing import Tuple, List
from numpy import sin, cos, tan, fabs, pi
import struct
# following this article: https://aishack.in/tutorials/sudoku-grabber-opencv-extracting-digits/

class DgtRecog:
    MAX_NUM_IMAGES = 60000
    def __init__(self):
        self.__numRows, self.__numCols, self.__numImages = 0, 0, 0
        self.__knn = cv2.ml.KNearest_create()

    def train(self, trainPath: str, labelsPath: str) -> bool:
        fp = None
        fp2 = None
        try:
            fp = open(file = trainPath, mode = "rb")
            fp2 = open(file = labelsPath, mode = "rb")
        except:
            print("Failed to train")
            return False

        magicNumber = self.__readFlippedInteger(fp)
        self.__numImages = self.__readFlippedInteger(fp)
        self.__numRows = self.__readFlippedInteger(fp)
        self.__numCols = self.__readFlippedInteger(fp)
        
        # Read 8 bytes from the start of the file
        fp2.seek(8) # skip first 8 bytes

        if self.__numImages > DgtRecog.MAX_NUM_IMAGES:
            self.__numImages = DgtRecog.MAX_NUM_IMAGES
        size = self.__numRows * self.__numCols

        trainingVectors = np.zeros((self.__numImages, size), dtype = 'float32')
        trainingClasses = np.zeros((self.__numImages, 1), dtype = 'float32')


        for i in range(self.__numImages):
            
            temp = bytearray(size)
            temp = fp.read(size)

            tempClass = bytearray(4)

            tempClass = int.from_bytes(fp2.read(1), 'big')
            trainingClasses[i] = tempClass

            for k in range(size):
                trainingVectors[i][k] = temp[k]

        print('training')
        self.__knn.train(trainingVectors,cv2.ml.ROW_SAMPLE, trainingClasses)
        print('trained')

        fp.close()
        fp2.close()

    def classify(self, img) -> int:
        cloneImg = self.__preprocessImage(img)
        cloneImg = cloneImg.astype(dtype = 'float32')
        return self.__knn.findNearest(cloneImg, 1)

    def __preprocessImage(self, img):

        rowTop = rowBot = colLeft = colRight = -1
        thresholdBot = thresholdTop = thresholdLeft = thresholdRight = 50
        temp = None
        center = len(img) // 2
        numRows = len(img)
        numCols = len(img[0])
        for i in range(center, len(img)):
            if rowBot == -1:
                temp = img[i]
                if cv2.sumElems(temp)[0] < thresholdBot or i == numRows - 1:
                    rowBot = i
            if rowTop == -1:
                temp = img[len(img) - i]
                if cv2.sumElems(temp)[0] < thresholdTop or i == numRows - 1:
                    rowTop = len(img) - i
            if colRight == -1:
                temp = img[:, i]
                if cv2.sumElems(temp)[0] < thresholdRight or i == numCols - 1:
                    colRight = i
            
            if colLeft == -1:
                temp = img[:, len(img[0]) - i]
                if cv2.sumElems(temp)[0] < thresholdLeft or i == numCols - 1:
                    colLeft = len(img[0]) - i
        
        newImg = np.zeros((numRows, numCols), dtype="uint8")

        startAtX = numRows // 2 - (colRight - colLeft) // 2
        startAtY = numCols // 2 - (rowBot - rowTop) // 2
        for y in range(startAtY, len(newImg)//2 + (rowBot - rowTop)//2):
            ptr = newImg[y]

            for x in range(startAtX, len(newImg[0])//2 + (colRight - colLeft)//2):
                ptr[x] = img[rowTop+(y-startAtY)][colLeft+(x-startAtX)]
        

        cloneImg = cv2.resize(newImg, (self.__numCols, self.__numRows))
        for y in range(self.__numRows):
            for x in range(self.__numCols):
                if y < 3 or y + 3> self.__numRows or x < 3 or x + 3 > self.__numCols:
                    cloneImg[y][x] = 0
        cv2.imwrite("number.png", cloneImg)

        for i in range(len(cloneImg)):
            cv2.floodFill(cloneImg, None, (0, i), (0, 0, 0, 0))
            cv2.floodFill(cloneImg, None, (len(cloneImg[0]) - 1, i), (0, 0, 0, 0))
            cv2.floodFill(cloneImg, None, (i, 0), (0, 0, 0, 0))
            cv2.floodFill(cloneImg, None, (i, len(cloneImg) - 1), (0, 0, 0, 0))

        cloneImg = np.reshape(cloneImg, (1, len(cloneImg)*len(cloneImg[0])))
        return cloneImg

    def __readFlippedInteger(self, fp) -> int:
        ret = 0

        word = bytearray(4)
        word = fp.read(4)
        ret = int.from_bytes(word, byteorder='big')

        print(ret)

        return ret
    def test(self, trainImgs, trainLabels, testImgs, testLabels):
        self.train(trainImgs, trainLabels)
        fp = None
        fp2 = None
        try:
            fp = open(file=testImgs, mode="rb")
            fp2 = open(file=testLabels, mode="rb")
        except:
            print("Failed to test")
            return False

        magicNumber = self.__readFlippedInteger(fp)
        numImages = self.__readFlippedInteger(fp)
        numRows = self.__readFlippedInteger(fp)
        numCols = self.__readFlippedInteger(fp)
        size = numRows * numCols

        # Read 8 bytes from the start of the file
        fp2.seek(8)  # skip first 8 bytes

        testVectors = np.zeros((numImages, size), dtype="float32")
        testLabels = np.zeros((numImages, 1), dtype="float32")
        actualLabels = np.zeros((numImages, 1), dtype="float32")

        temp = bytes(size)
        tempClass = 1
        currentTest = np.zeros((1, size), dtype='float32')
        currentLabel = np.zeros((1, 1), dtype="float32")
        totalCorrect = 0
        for i in range(numImages):
            temp = bytearray(size)
            temp = fp.read(size)

            tempClass = fp2.read(1)

            actualLabels[i][0] = int.from_bytes(tempClass, 'big')

            for k in range(size):
                testVectors[i][k] = temp[k]
                currentTest[0][k] = temp[k]

            self.__knn.findNearest(currentTest, 5, currentLabel)
            testLabels[i][0] = currentLabel[0][0]

            if currentLabel[0][0] == actualLabels[i][0]:
                totalCorrect += 1
        fp.close()
        fp2.close()
        print("Accuracy: {} ".format( totalCorrect*100/numImages))



if __name__ == "__main__":
    dr = DgtRecog()
    dr.test("train-images-idx3-ubyte","train-labels-idx1-ubyte",\
            "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte",)
    print('ree')
