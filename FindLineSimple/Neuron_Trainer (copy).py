import numpy as numpyLib
import cv2
import math
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt


def getGradientPoints(image):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(img_grey, cv2.CV_16S, 1, 0)
    abs_dx = cv2.convertScaleAbs( dx)
    dy = cv2.Sobel(img_grey, cv2.CV_16S, 0, 1)
    abs_dy = cv2.convertScaleAbs( dy)
    grad_approx = cv2.addWeighted( abs_dx, 0.5, abs_dy, 0.5, 0 )
    #cv2.namedWindow("gradient", cv2.CV_WINDOW_AUTOSIZE)
    #cv2.imshow("gradient",grad_approx)
    #cv2.waitKey(0) & 0xFF
    #cv2.destroyWindow("gradient")
    averageValue = numpyLib.average(grad_approx)
    nonZeroPoints = numpyLib.nonzero(grad_approx > averageValue * 0.95)

    return [(a[1], a[0]) for a in zip(*nonZeroPoints)]

def getPointsFromImage(fileName, size_of_ann, rotation_degrees = 0):
    img = cv2.imread(fileName, cv2.CV_LOAD_IMAGE_COLOR)

    gradientPoints = getGradientPoints(img)
    # Initiate STAR detector
    orb = cv2.ORB(1800)

    # find the keypoints with ORB
    keypoints = orb.detect(img,None)

    orbPoints = [x.pt for x in keypoints]

    mergedPoints = orbPoints + gradientPoints

    # compute the descriptors with ORB
    #kp, des = orb.compute(img, kp)
    mid_x = 0
    mid_y = 0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    if len(mergedPoints) > 0:
        print "path: ", fileName
        print "points found: ", len(mergedPoints)
        rows = img.shape[0]
        cols = img.shape[1]
        print "rows: ", rows, " cols: ", cols
        mid_x = 0
        mid_y = 0
        min_x = mergedPoints[0][0]
        min_y = mergedPoints[0][1]
        max_x = min_x
        max_y = min_y
        for point in mergedPoints:
            x = point[0]
            y = point[1]
            mid_x += x
            mid_y += y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        mid_x /= len(mergedPoints)
        mid_y /= len(mergedPoints)
    max_dist_from_center = max(mid_x - min_x, max_x - mid_x, mid_y - min_y, max_y - mid_y)
    ratio = size_of_ann / (2.0 * max_dist_from_center)
    min_x_i = math.floor(min_x).__int__()
    min_y_i = math.floor(min_y).__int__()
    max_x_i = math.floor(max_x).__int__()
    max_y_i = math.floor(max_y).__int__()
    # draw only keypoints location,not size and orientation
    img2 = img.copy()
    cv2.line(img2, (min_x_i, min_y_i), (min_x_i, max_y_i), (0, 255, 0), 4)
    cv2.line(img2, (min_x_i, min_y_i), (max_x_i, min_y_i), (0, 255, 0), 4)
    cv2.line(img2, (max_x_i, max_y_i), (max_x_i, min_y_i), (0, 255, 0), 4)
    cv2.line(img2, (min_x_i, max_y_i), (max_x_i, max_y_i), (0, 255, 0), 4)

    result_vector = numpyLib.zeros((size_of_ann , size_of_ann))



    for point in mergedPoints:
        x = point[0]
        y = point[1]
        x -= mid_x
        y -= mid_y
        x *= ratio
        y *= ratio
        x += size_of_ann / 2
        y += size_of_ann / 2
        ix = x.__int__()
        iy = y.__int__()
        ix = min(size_of_ann - 1, ix)
        iy = min(size_of_ann - 1, iy)
        ix = max(0, ix)
        iy = max(0, iy)
        cv2.circle(img2, (ix, iy), 1, (0, 255, 0))
        cv2.circle(
             img2,
             (point[0].__int__(), point[1].__int__()),
             1,
             (255, 0, 0))
        #index = math.floor(y * size_of_ann + x).__int__()
        #index = min(index, size_of_ann * size_of_ann)
        #index = max(index, 0)
        result_vector[iy, ix] = 1
    cv2.namedWindow("FoundKeyPoints", cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("FoundKeyPoints", img2)
    cv2.waitKey(0) & 0xFF
    cols = size_of_ann
    rows = size_of_ann
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2),rotation_degrees,1)
    dst = cv2.warpAffine(result_vector,M,(cols,rows))
    result_vector = dst.copy()
    cv2.imshow("FoundKeyPoints", result_vector)
    cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("FoundKeyPoints")
    return result_vector.flatten('C')


imagesNames = [f for f in listdir("./") if isfile(join("./", f)) and
    (f.endswith(".png") or f.endswith(".jpg"))]
learningImages = [f for f in imagesNames
                    if f.startswith("success_") or
                         f.startswith("failure_")]

print imagesNames

size_of_ann = 40
count_of_images = len(learningImages)
print count_of_images





# Create train samples
# train sum function

# Create network with 2 inputs, 5 neurons in input layer
# And 1 in output layer
from ffnet import loadnet
print("Loadting artificial neuron network:")
net = loadnet("line.net")
print("Starting testing:")
# Test
for imagePath in imagesNames:
    for degree in range(0, 180, 10) :
        res = net(
            getPointsFromImage(imagePath, size_of_ann, degree).reshape(1, size_of_ann * size_of_ann))
        print imagePath," result: ", res, " degree: ",degree


