import numpy as numpyLib
import cv2
import math
from os import listdir
from os.path import isfile, join



def getPointsForAnn(keyPointsOfImage,
        size_of_ann,
        verbose=False,
        verboseOriginalImage=None):

    mergedPoints = keyPointsOfImage
    mid_x = 0
    mid_y = 0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    ratio = 1.0
    max_dist_from_center = 1
    if len(mergedPoints) > 0:
        print "points found: ", len(mergedPoints)
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
        max_dist_from_center = max(
                                mid_x - min_x,
                                max_x - mid_x,
                                mid_y - min_y,
                                max_y - mid_y)
        # for case when there is only one point we don't need to zoom it
        if max_dist_from_center == 0:
            ratio = 1
        else:
            # if bounding rectangle around points can fit into ANN
            # we also calucalate ratio
            if 2 * max_dist_from_center < size_of_ann:
                ratio = size_of_ann / (2.0 * max_dist_from_center)
            else:
                ratio = size_of_ann / (2.0 * max_dist_from_center)

    result_vector = numpyLib.zeros((size_of_ann, size_of_ann))
    img2 = None
    if verbose:
        min_x_i = math.floor(min_x).__int__()
        min_y_i = math.floor(min_y).__int__()
        max_x_i = math.floor(max_x).__int__()
        max_y_i = math.floor(max_y).__int__()
        img2 = verboseOriginalImage.copy()
        cv2.line(img2, (min_x_i, min_y_i), (min_x_i, max_y_i), (0, 255, 0), 4)
        cv2.line(img2, (min_x_i, min_y_i), (max_x_i, min_y_i), (0, 255, 0), 4)
        cv2.line(img2, (max_x_i, max_y_i), (max_x_i, min_y_i), (0, 255, 0), 4)
        cv2.line(img2, (min_x_i, max_y_i), (max_x_i, max_y_i), (0, 255, 0), 4)

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
        if verbose:
            #draw key points locations
            cv2.circle(img2, (ix, iy), 1, (0, 255, 0))
            cv2.circle(
                 img2,
                 (point[0].__int__(), point[1].__int__()),
                 1,
                 (255, 0, 0))
        result_vector[iy, ix] = 1
    if verbose:
        cv2.namedWindow("FoundKeyPoints", cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("FoundKeyPoints", img2)
        cv2.waitKey(0) & 0xFF
    min_x_i = math.floor(min_x).__int__()
    min_y_i = math.floor(min_y).__int__()
    max_x_i = math.floor(max_x).__int__()
    max_y_i = math.floor(max_y).__int__()
    return (result_vector , (min_x_i, min_y_i), (max_x_i, max_y_i))


def applyANN(size_of_ann,
         ann_net,
         result_vector,
         rotation_degrees=0,
         verbose=False):
    cols = size_of_ann
    rows = size_of_ann
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_degrees, 1)
    temp_vector = cv2.warpAffine(result_vector, M, (cols, rows))
    if verbose:
        cv2.imshow("FoundKeyPoints", temp_vector)
        cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("FoundKeyPoints")
    return ann_net(temp_vector.
        flatten('C').
        reshape(1, size_of_ann * size_of_ann))


def subAnalyzeImage(keyPoints, size_of_ann, ann_net,
        verbose=False, splitPointsInFourParts=True):
    topLeftPoint = None
    bottomRightPoint = None
    analyzedObjects = []
    if len(keyPoints) == 0:
        analyzedObjects.append(
                ("non_determined",
                (1.0, 0.0),
                topLeftPoint,
                bottomRightPoint,
                keyPoints
                )
            )
        return analyzedObjects
    else:
        results = []
        pointsForAnnTuple = getPointsForAnn(keyPoints, size_of_ann)
        (pointsForAnn, topLeftPoint, bottomRightPoint) = pointsForAnnTuple
        (columns_start, rows_start) = topLeftPoint
        (columns_end, rows_end) = bottomRightPoint
        columns = columns_end - columns_start + 1
        rows = rows_end - rows_start + 1
        for degree in range(0, 360, 4) :
            res = applyANN(size_of_ann, ann_net, pointsForAnn, degree)
            results.append((res, degree))
            if verbose:
                print imagePath," result: ", res, " degree: ",degree
        maxResult = max(results,key=lambda (result,_): result)
        (result, degree) = maxResult
        if result > 0.5:
            print "accepted as line result: ", result, " degree: ",degree
            analyzedObjects.append(
                    ("line",
                    maxResult,
                    topLeftPoint,
                    bottomRightPoint,
                    keyPoints
                    )
                )
            return analyzedObjects
        else:
            if rows < 5 and columns < 5:
                analyzedObjects.append(
                        ("non_determined",
                        maxResult,
                        topLeftPoint,
                        bottomRightPoint,
                        keyPoints
                        )
                    )
                return analyzedObjects
            else:
                if splitPointsInFourParts:
                    print "Splitting points in four parts"
                    columns_mid = (columns_start + columns_end) / 2
                    rows_mid = (rows_start + rows_end) / 2
                    points_top_left = [point for point in keyPoints
                        if point[0] < columns_mid and
                            point[1] < rows_mid]
                    results_top_left = subAnalyzeImage(points_top_left, size_of_ann,
                         ann_net,
                         verbose)
                    analyzedObjects = results_top_left + analyzedObjects
                    points_top_right = [point for point in keyPoints
                        if point[0] >= columns_mid and
                            point[1] < rows_mid]
                    results_top_right = subAnalyzeImage(points_top_right,
                        size_of_ann,
                        ann_net,
                        verbose)
                    analyzedObjects = results_top_right + analyzedObjects
                    points_bottom_left = [point for point in keyPoints
                        if point[0] < columns_mid and
                            point[1] >= rows_mid]
                    results_bottom_left = subAnalyzeImage(points_bottom_left,
                         size_of_ann,
                         ann_net,
                         verbose)
                    analyzedObjects = results_bottom_left + analyzedObjects
                    points_bottom_right = [point for point in keyPoints
                        if point[0] >= columns_mid and
                            point[1] >= rows_mid]
                    results_bottom_right = subAnalyzeImage(points_bottom_right,
                        size_of_ann,
                        ann_net,
                        verbose)
                    analyzedObjects = results_bottom_right + analyzedObjects
                return analyzedObjects


def drawAnalyzedResults(image, results):
    img2 = image.copy()
    for item in results:
        (name,
            (probability, degree),
            top_left_point,
            bottom_right_point,
            keyPoints) = item
        if name == "line":
            cv2.rectangle(img2,
                top_left_point,
                bottom_right_point,
                (200, 0, 0)
                )
            print degree
    cv2.namedWindow("AnalyzedObjects", cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("AnalyzedObjects", img2)
    cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("AnalyzedObjects")


def tryToCombineLines(results, size_of_ann, ann_net):
    onlyLines = [x for x in results if x[0] == "line"]
    enumeratedLines = enumerate(onlyLines)
    result = []
    for indexLeft, leftValue in enumeratedLines:
        temp = []
        for indexRight, rightValue in enumeratedLines:
            (_,
                (probability_left, degree_left),
                _,
                _,
                keyPointsLeft) = leftValue
            (_,
                (probability_right, degree_right),
                _,
                _,
                keyPointsRight) = rightValue
            abs_diff = abs(degree_left - degree_right)
            alpha = 10
            abs_180 = abs(180 - abs_diff)
            if abs_diff < alpha or abs_180 < alpha:
                mergedPoints = keyPointsLeft + keyPointsRight
                combinedResult = subAnalyzeImage(mergedPoints,
                    size_of_ann, ann_net,
                    splitPointsInFourParts=False)
                if len(combinedResult) == 1:
                    temp = combinedResult + temp
        result = max(temp, lambda x: x[1][0]) + result

    return result




imagesNames = [f for f in listdir("./") if isfile(join("./", f)) and
    (f.endswith(".png") or f.endswith(".jpg"))]
learningImages = [f for f in imagesNames
                    if f.startswith("success_") or
                         f.startswith("failure_")]

print imagesNames



size_of_ann = 40
count_of_images = len(learningImages)
print count_of_images

import pickle
from ffnet import loadnet
print("Loading artificial neuron network:")
net = loadnet("line.net")
print("Starting testing:")
# Test
for imagePath in imagesNames:
    print "path: ", imagePath
    image = cv2.imread(imagePath, cv2.CV_LOAD_IMAGE_COLOR)
    analyzedObjects = []
    with open(imagePath + '_analyzedObjects.txt', 'rb') as f:
        analyzedObjects = pickle.load(f)
    #print "Analyze result: ", analyzedObjects
    drawAnalyzedResults(image, results=analyzedObjects)
    combinedObjects = tryToCombineLines(analyzedObjects, size_of_ann, net)
    drawAnalyzedResults(image, results=combinedObjects)



