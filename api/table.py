import cv2
import numpy as np
from typing import Tuple, List
from numpy import sin, cos, tan, fabs, pi
import extract

# following this article: https://aishack.in/tutorials/sudoku-grabber-opencv-detection/
def segment():

    # TODO downscale to 1080p if greater
    #imgName = "sudoku_4k.png"
    imgName = "sudoku_test.png"
    #imgName = "sudoku-original.jpg"
    # import np array of image
    sudoku = cv2.imread(imgName, 0)

    # empty box to fit image
    outbox = np.zeros((len(sudoku), len(sudoku[0])), dtype = "uint8")

    # gaussian blur algo magic to thicc the lines for easier recogntion
    sudoku = cv2.GaussianBlur(sudoku, (11, 11), 0)

    # see thresholding - even out lighting
    outbox = cv2.adaptiveThreshold(sudoku, 255,  cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 2)
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

    # TODO erosion - hough transform - minor gridline calculations
    # see https://aishack.in/tutorials/sudoku-grabber-opencv-detection/
    outbox = cv2.erode(outbox, kernel)
    print(cv2.imwrite("gridlines.png", outbox))


    # CV_PI = 3.1415926535897932384626433832795
    lines = cv2.HoughLines(outbox, 1, pi/180, 200)
    houghlines = np.zeros((len(outbox), len(outbox[0])), dtype = "uint8")
    # debug: print houghlines
    for line in lines:
        drawLine(line, houghlines, (128, 0, 0, 0))
    print(cv2.imwrite("houghLines.png", houghlines))

    mergedLines = np.zeros((len(outbox), len(outbox[0])), dtype = "uint8")
    lines = mergeRelatedLines(lines, mergedLines)
   # lines = mergeRelatedLines(lines, mergedLines)

    for line in lines:
        drawLine(line, mergedLines, (128, 0, 0, 0))
    print(cv2.imwrite("mergeHough.png", mergedLines))


    # find extreme lines
    finalOutbox = findExtremeLines(lines, sudoku)
    print(cv2.imwrite("undistort.png", finalOutbox))
    return finalOutbox

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
    
def drawLine(line, img, rgb = (255, 0, 0, 0)):
    line = line[0]
    rho = line[0]
    theta = line[1]
    
    if theta != 0:
        m = -1/tan(theta)
        c = rho/sin(theta)
        #print("current r: {}, theta: {}, m: {}, c: {}".format(line[0], line[1], m, c))
        newLine = cv2.line(img, (0, np.int_(c)), (len(img[0]), np.int_(m * len(img[0]) + c)), rgb)
    else:
        newLine = cv2.line(img, (np.int_(rho), 0), (np.int_(rho), len(img)), rgb)

    return True

# we have no idea what we're doing here hahahahahahahahaha
def mergeRelatedLines(lines, img):

    for line1 in lines:
        curr = line1[0]
        p1 = curr[0]
        theta1 = curr[1]
        if np.int_(p1) == 0 and np.int_(theta1) == -100:
            continue
            
        pt1curr, pt2curr = None, None
        
        if theta1 > pi*45/180 and theta1 < pi*135/180:
            pt1curr = (0, np.int_(p1/sin(theta1)))
            pt2curr = (len(img[0]), \
                       np.int_(-len(img[0])/tan(theta1) + p1/sin(theta1)))
        else:
            pt1curr = (p1/cos(theta1), 0)
            pt2curr = (np.int_(-len(img)*tan(theta1) + p1/cos(theta1)),\
                    len(img))
            
        for line2 in lines:
            pos = line2[0]
            if pos[0] == curr[0] and pos[1] == curr[1]:
                continue

            #scaling issues?

            if fabs(pos[0] - curr[0]) < 20 and fabs(pos[1] - curr[1]) < pi*10/180:
                p = pos[0]
                theta = pos[1]
                pt1, pt2 = None, None
                if pos[1] > pi*45/180 and pos[1] < pi * 135/180:
                    pt1 = (0, np.int_(p / sin(theta)))
                    pt2 = (len(img[0]), np.int_(-len(img[0])/tan(theta) + p/sin(theta)))
                else:
                    pt1 = (np.int_(p/cos(theta)), 0)
                    pt2 = (np.int_(-len(img)*tan(theta) + p/cos(theta)), len(img))
                #scaling issues?
                if (pt1[0] - pt1curr[0])**2 + (pt1[1] - pt1curr[1])**2 < 64**2 and \
                    (pt2[0] - pt2curr[0])**2 + (pt2[1] - pt2curr[1])**2 < 64**2:
                    curr = (int((curr[0] + pos[0]) / 2), int((curr[1] + pos[1]) / 2))
                    line2[0] = [0, -100]

    return lines

def findExtremeLines(lines, original):
    # edges are lines in polar form. Temporarily garbage values
    topEdge = (0, 0)
    botEdge = (10000, 10000)
    leftEdge = (10000, 10000)
    rightEdge = (0, 0)

    for line in lines:
        current = line[0]
        p = current[0]
        theta = current[1]
        #print(line)
        if p == 0 and theta == -100:
            continue

        if theta > pi*80/180 and theta < pi*100/180:
            if fabs(p) > fabs(topEdge[0]):
                topEdge = current
            
            if fabs(p) < fabs(botEdge[0]):
                botEdge = current


        elif theta < pi*10/180 or theta > pi*170/180:
            if fabs(p) < fabs(leftEdge[0]):
                leftEdge = current


            if fabs(p) > fabs(rightEdge[0]):
                rightEdge = current

        

        # TODO if wanted, we can draw the extremes now for reference 
        # return [topEdge, botEdge, leftEdge, rightEdge]
    
    print(leftEdge, rightEdge, topEdge, botEdge)
    edgeLines = np.zeros((len(original), len(original[0])), dtype = "uint8")
    for line in [[leftEdge], [rightEdge], [topEdge], [botEdge]]:
        drawLine(line, edgeLines, (128, 0, 0, 0))
    print(cv2.imwrite("edgeLines.png", edgeLines))
        
    # points on the edges
    l1 = l2 = r1 = r2 = b1 = b2 = t1 = t2 = (0,0)

    imgH = len(original)
    imgW = len(original[0])
    print(imgH, imgW)
    
    if leftEdge[1] != 0:
        l1 = (0, leftEdge[0]/sin(leftEdge[1]))
        l2 = (imgW, -imgW/tan(leftEdge[1]) + l1[1])
    else:
        l1 = (leftEdge[0]/cos(leftEdge[1]), 0)
        l2 = (l1[0] - imgH*tan(leftEdge[1]), imgH)

    if rightEdge[1] != 0:
        r1 = (0, rightEdge[0]/sin(rightEdge[1]))
        r2 = (imgW, -imgW/tan(rightEdge[1]) + r1[1])
    else:
        r1 = (rightEdge[0]/cos(rightEdge[1]), 0)
        r2 = (r1[0] - imgH*tan(rightEdge[1]), imgH)

    b1 = (0, botEdge[0]/sin(botEdge[1]))
    b2 = (imgW, -imgW/tan(botEdge[1]) + b1[1])
    
    t1 = (0, topEdge[0]/sin(topEdge[1]))
    t2 = (imgW, -imgW/tan(topEdge[1]) + t1[1])


    # find points of intersection using line-line intersection method
    # see the following for more: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    # https://aishack.in/tutorials/solving-intersection-lines-efficiently/
    lA = lB = lC = rA = rB = rC = bA = bB = bC = tA = tB = tC = 0

    lA = l2[1]-l1[1] 
    lB = l1[0]-l2[0] 
    lC = lA*l1[0] + lB*l1[1]

    rA = r2[1]-r1[1]
    rB = r1[0]-r2[0]
    rC = rA*r1[0] + rB*r1[1]
    
    tA = t2[1]-t1[1]
    tB = t1[0]-t2[0]
    tC = tA*t1[0] + tB*t1[1]
    
    bA = b2[1]-b1[1]
    bB = b1[0]-b2[0]
    bC = bA*b1[0] + bB*b1[1]

    #  mat = a b  =  lA lB
    #        c d     tA tB 
    #  det = ad - bc
    #  det = lA*tB - lB*tA
    detTopLeft = lA*tB - lB*tA
    ptTopLeft = (np.float32((tB*lC - lB*tC)/detTopLeft), np.float32((lA*tC - tA*lC)/detTopLeft))

    detTopRight = rA*tB - rB*tA
    ptTopRight = (np.float32((tB*rC-rB*tC)/detTopRight), np.float32((rA*tC-tA*rC)/detTopRight))

    detBottomRight = rA*bB - rB*bA
    ptBottomRight = (np.float32((bB*rC-rB*bC)/detBottomRight), np.float32((rA*bC-bA*rC)/detBottomRight))

    detBottomLeft = lA*bB-lB*bA
    ptBottomLeft = (np.float32((bB*lC-lB*bC)/detBottomLeft), np.float32((lA*bC-bA*lC)/detBottomLeft))

    # find longest edge, create new image with square of longest edge
    maxLength = (ptBottomLeft[0]-ptBottomRight[0])*(ptBottomLeft[0]-ptBottomRight[0]) + (ptBottomLeft[1]-ptBottomRight[1])*(ptBottomLeft[1]-ptBottomRight[1])
    temp = (ptTopRight[0]-ptBottomRight[0])*(ptTopRight[0]-ptBottomRight[0]) + (ptTopRight[1]-ptBottomRight[1])*(ptTopRight[1]-ptBottomRight[1])

    if temp > maxLength:
        maxLength = temp

    temp = (ptTopRight[0]-ptTopLeft[0])*(ptTopRight[0]-ptTopLeft[0]) + (ptTopRight[1]-ptTopLeft[1])*(ptTopRight[1]-ptTopLeft[1])

    if temp > maxLength:
        maxLength = temp

    temp = (ptBottomLeft[0]-ptTopLeft[0])*(ptBottomLeft[0]-ptTopLeft[0]) + (ptBottomLeft[1]-ptTopLeft[1])*(ptBottomLeft[1]-ptTopLeft[1])

    if temp > maxLength:
        maxLength = temp
    maxLength = np.int_(np.sqrt(maxLength))

    # create new undistorted scaled with longest edge
    # prepare points for warping image
    src = np.array([ptBottomLeft, ptBottomRight, ptTopRight, ptTopLeft], dtype = "float32")

    dst = np.array([(0,0), (maxLength - 1, 0), (maxLength - 1, maxLength - 1), (0, maxLength - 1)], dtype = "float32")
    print(src)
    print(dst)
  
    # get undistorted image
    undistorted = cv2.warpPerspective(original, cv2.getPerspectiveTransform(src, dst), (maxLength, maxLength))

    return undistorted

def RecogDGT(img):
    maxLength = len(img)
    undistort = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 101, 1)
    kernel = np.array(((0,1,0), (1, 1, 1), (0, 1, 0)), dtype = "uint8")

    print(cv2.imwrite("recog.png", undistort))
    dr = extract.DgtRecog()
    b = dr.train("train-images-idx3-ubyte","train-labels-idx1-ubyte")
    dist = np.int_(np.ceil(maxLength/9))
    currentCell = np.zeros((dist, dist), dtype="uint8")

    for row in range(9):
        for col in range(9):
            y = 0
            while y < dist and row * dist + y < len(undistort):
                x = 0
                ptr = currentCell[y]
                while x < dist and col * dist + x < len(undistort[0]):
                    if y < 5 or y + 5> dist or x < 5 or x +5> dist:
                        x += 1
                        continue
                    
                    ptr[x] = undistort[row * dist + y][col * dist + x]
                    x += 1
                y += 1

            m = cv2.moments(currentCell, True)
            area = m['m00']

            if area > (len(currentCell) * len(currentCell[0]))//5:
                number = dr.classify(currentCell)
                print(number[0], end = '')
                input()
            else:
                print(" ", end='')
        print("")



if __name__ == "__main__":
    img = segment()
    RecogDGT(img)
    print('were done buddy')