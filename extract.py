import cv2
import pytesseract
import numpy as np

def classify(img) -> int:
    custom_config = r'--oem 3 --psm 10 outputbase digits'
    cloneImg = preprocessImage(img)
    return pytesseract.image_to_string(cloneImg, config = custom_config)

def preprocessImage(img):

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
    
    # pytesseract works best with digits of 25 - 40 pixels
    # reize to 32 x 32 pixel image
    finalRows = finalCols = 32
    cloneImg = cv2.resize(newImg, (finalRows, finalCols))
    cv2.imwrite("number.png", cloneImg)

    return cloneImg
