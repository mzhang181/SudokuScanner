import cv2
import numpy as np

# following this article: https://aishack.in/tutorials/sudoku-grabber-opencv-detection/
def segment():
    # import np array of image
    sudoku = cv2.imread("megaSudoku.bmp", 0)

    # empty box to fit image
    outbox = np.zeros((len(sudoku), len(sudoku[0])), dtype = "uint8")

    # gaussian blur algo magic to thicc the lines for easier recogntion
    sudoku = cv2.GaussianBlur(sudoku, (11, 11), 0)

    # see thresholding - even out lighting
    outbox = cv2.adaptiveThreshold(sudoku, 255,  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
    #print(cv2.imwrite("thresholdReee.png", outbox))

    # inverse image
    outbox = cv2.bitwise_not(outbox)

    # create pixel neighbourhood we wish to use for dilation - we used 4-neighbourhood
    #   0 1 0
    #   1 1 1   4-neighbourhood
    #   0 1 0
    kernel = np.array(((0,1,0), (1, 1, 1), (0, 1, 0)), dtype = "uint8")

    # iterate through pixels, check for given neighbourhood, set to according value
    outbox = cv2.dilate(outbox, kernel)

    #print(cv2.imwrite("filtered.png", outbox))

    outbox = findGridBlob(outbox)
    print(cv2.imwrite("gridlines.png", outbox))

    # TODO erosion - hough transform - minor gridline calculations
    # see https://aishack.in/tutorials/sudoku-grabber-opencv-detection/
    return outbox

# find major grid lines
def findGridBlob(outbox):
    maxArea = -1
    maxPt = None
    
    for y, row in enumerate(outbox):
        for x, value in enumerate(row):
            if value >= 128:
                area = cv2.floodFill(outbox, None, (x,y), (64,0,0, 0))
                if area[0] > maxArea:
                    maxPt = (x, y)
                    maxArea = area[0]
    area = cv2.floodFill(outbox, None, maxPt, (255,255,255, 0))
                    
    for y, row in enumerate(outbox):
        for x, value in enumerate(row):
            if value == 64 and x != maxPt[0] and y != maxPt[1]:
                area = cv2.floodFill(outbox, None, (x,y), (0,0,0,0))
                            
    return outbox
    

if __name__ == "__main__":
    segment()
   