import numpy as numpyLib
import cv2
import math
from os import listdir
from os.path import isfile, join
import timeit
import sys
import skfuzzy as fuzz


def getGradientPointsFromImage(image, verbose=False):
    img_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dx = cv2.Sobel(img_grey, cv2.CV_16S, 1, 0)
    abs_dx = cv2.convertScaleAbs(dx)
    dy = cv2.Sobel(img_grey, cv2.CV_16S, 0, 1)
    abs_dy = cv2.convertScaleAbs(dy)
    grad_approx = cv2.addWeighted(abs_dx, 0.5, abs_dy, 0.5, 0)
    if verbose:
        cv2.namedWindow("gradient", cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("gradient", grad_approx)
    averageValue = numpyLib.average(grad_approx)
    nonZeroPoints = numpyLib.nonzero(grad_approx > averageValue * 0.95)

    return [(a[1], a[0]) for a in zip(*nonZeroPoints)]


def getKeyPoints(image, getGradientPoints=True, getORBKeyPoints=True):
    img = image
    mergedPoints = []
    if getGradientPoints:
        print( "Calculating gradient points")
        gradientPoints = getGradientPointsFromImage(img, verbose=True)
        mergedPoints = gradientPoints + mergedPoints
    if getORBKeyPoints:
        print "Calculating ORB points"
        # Initiate STAR detector
        orb = cv2.ORB(1800)
        # find the keypoints with ORB
        keypoints = orb.detect(img, None)
        orbPoints = [x.pt for x in keypoints]
        # compute the descriptors with ORB
        #kp, des = orb.compute(img, kp)
        mergedPoints = orbPoints + mergedPoints
    return mergedPoints

def getBoundingRectangle(points):
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    if len(points) > 0:
        min_x = points[0][0]
        min_y = points[0][1]
        max_x = min_x
        max_y = min_y
        for point in points:
            x = point[0]
            y = point[1]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return ((min_x, min_y), (max_x, max_y))

def enlargeBoundingRectangle(points, min_x, min_y, max_x, max_y):
    if len(points) > 0:
        for point in points:
            x = point[0]
            y = point[1]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    return ((min_x, min_y), (max_x, max_y))



def getPointsForAnn(keyPointsOfImage,
        size_of_ann,
        verbose=False,
        verboseOriginalImage=None):

    mergedPoints = keyPointsOfImage
    mid_x = 0
    mid_y = 0
    mass_center_x = 0
    mass_center_y = 0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    ratio = 1.0
    max_dist_from_center = 1
    if len(mergedPoints) > 0:
        mid_x = 0
        mid_y = 0
        min_x = mergedPoints[0][0]
        min_y = mergedPoints[0][1]
        max_x = min_x
        max_y = min_y
        for point in mergedPoints:
            x = point[0]
            y = point[1]
            mass_center_x += x
            mass_center_y += y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        mass_center_x /= len(mergedPoints)
        mass_center_y /= len(mergedPoints)
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0
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
        cv2.line(img2, (0, 0), (size_of_ann, 0), (0, 0, 255), 1)
        cv2.line(img2, (0, 0), (0, size_of_ann), (0, 0, 255), 1)
        cv2.line(img2, (0, size_of_ann), (size_of_ann, size_of_ann), (0, 0, 255), 1)
        cv2.line(img2, (size_of_ann, 0), (size_of_ann, size_of_ann), (0, 0, 255), 1)
        cv2.namedWindow("FoundKeyPoints", cv2.CV_WINDOW_AUTOSIZE)
        cv2.imshow("FoundKeyPoints", img2)
        cv2.waitKey(0) & 0xFF
    min_x_i = math.floor(min_x).__int__()
    min_y_i = math.floor(min_y).__int__()
    max_x_i = math.floor(max_x).__int__()
    max_y_i = math.floor(max_y).__int__()
    return (result_vector , (min_x_i, min_y_i), (max_x_i, max_y_i))

def rotatePoints(keyPoints, degree):
    theta = numpyLib.radians(degree)
    c, s = numpyLib.cos(theta), numpyLib.sin(theta)
    R = numpyLib.matrix('{} {}; {} {}'.format(c, s, -s, c))
    points = numpyLib.matrix(keyPoints)
    result = (points * R ).tolist()
    return result

def applyANN(size_of_ann,
         ann_net,
         result_vector,
         rotation_degrees=0,
         verbose=False):
    cols = size_of_ann
    rows = size_of_ann
    #M = cv2.getRotationMatrix2D((cols / 2, rows / 2), rotation_degrees, 1)
    #temp_vector = cv2.warpAffine(result_vector, M, (cols, rows))
    temp_vector = result_vector
    if verbose:
        cv2.imshow("FoundKeyPoints", temp_vector)
        cv2.waitKey(0) & 0xFF
        cv2.destroyWindow("FoundKeyPoints")
    return ann_net(temp_vector.
        flatten('C').
        reshape(1, size_of_ann * size_of_ann))



def getAngleOfLine(lineObject):
    (pointA, pointB) = lineObject
    (x, y) = pointA
    (x2, y2) = pointB
    vX = x2 - x
    vY = y2 - y
    norm = EuclidianNorm((vX, vY))
    if norm == 0:
        return 0
    vX /= norm
    #vY /= norm
    angleRad = math.acos(vX)
    angleDegree = math.degrees(angleRad)
    if vY < 0 :
        angleDegree = -angleDegree
    return angleDegree


def subAnalyzeImageForLine(keyPoints, size_of_ann, ann_net,
        verbose=False, splitPointsInFourParts=True, originalImage=None):
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
        pointsForAnnTuple = getPointsForAnn(keyPoints, size_of_ann, verbose=False, verboseOriginalImage=originalImage)
        (pointsForAnn, topLeftPoint, bottomRightPoint) = pointsForAnnTuple
        (columns_start, rows_start) = topLeftPoint
        (columns_end, rows_end) = bottomRightPoint
        columns = columns_end - columns_start + 1
        rows = rows_end - rows_start + 1
        lineObject = getLineObject(keyPoints)
        lineAngle = getAngleOfLine(lineObject)
        lineAngle = -lineAngle
        for degree in numpyLib.linspace(lineAngle - 5, lineAngle + 5, 10) :
            rotatedPoints = rotatePoints(keyPoints, degree)
            (rotatedPointsForAnn, _, _) = getPointsForAnn(rotatedPoints, size_of_ann, verbose=False, verboseOriginalImage=originalImage)
            res = applyANN(size_of_ann, ann_net, rotatedPointsForAnn, degree, verbose=False)
            results.append((res, degree))
            if verbose:
                print imagePath," result: ", res, " degree: ",degree
        maxResult = max(results,key=lambda (result,_): result)
        (result, degree) = maxResult
        if verbose:
            print "Result: ", result, " degree: ",degree
            img2 = originalImage.copy()
            for point in keyPoints:
                cv2.circle(
                     img2,
                     (point[0].__int__(), point[1].__int__()),
                     1,
                     (0, 255, 0))
            cv2.rectangle(
                   img2,
                   topLeftPoint,
                   bottomRightPoint,
                   (0, 0, 255)
                )
            cv2.namedWindow("PointsToAnalyze", cv2.CV_WINDOW_AUTOSIZE)
            cv2.imshow("PointsToAnalyze", img2)
            cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("PointsToAnalyze")
        if result >= 0.7:
            #print "accepted as line result: ", result, " degree: ",degree
            print '*',
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
                    print "Not a line, result: ", result, " degree: ",degree
                    print "Splitting points in four parts"
                    columns_mid = (columns_start + columns_end) / 2
                    rows_mid = (rows_start + rows_end) / 2
                    points_top_left = [point for point in keyPoints
                        if point[0] < columns_mid and
                            point[1] < rows_mid]
                    results_top_left = subAnalyzeImage(points_top_left, size_of_ann,
                         ann_net,
                         verbose,
                         originalImage=originalImage)
                    analyzedObjects = results_top_left + analyzedObjects
                    points_top_right = [point for point in keyPoints
                        if point[0] >= columns_mid and
                            point[1] < rows_mid]
                    results_top_right = subAnalyzeImage(points_top_right,
                        size_of_ann,
                        ann_net,
                        verbose,
                        originalImage=originalImage)
                    analyzedObjects = results_top_right + analyzedObjects
                    points_bottom_left = [point for point in keyPoints
                        if point[0] < columns_mid and
                            point[1] >= rows_mid]
                    results_bottom_left = subAnalyzeImage(points_bottom_left,
                         size_of_ann,
                         ann_net,
                         verbose,
                         originalImage=originalImage)
                    analyzedObjects = results_bottom_left + analyzedObjects
                    points_bottom_right = [point for point in keyPoints
                        if point[0] >= columns_mid and
                            point[1] >= rows_mid]
                    results_bottom_right = subAnalyzeImage(points_bottom_right,
                        size_of_ann,
                        ann_net,
                        verbose,
                        originalImage=originalImage)
                    analyzedObjects = results_bottom_right + analyzedObjects
                return analyzedObjects


def subAnalyzeImage(keyPoints, size_of_ann, ann_net,
        verbose=False, splitPointsInFourParts=True, originalImage=None):
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
        pointsForAnnTuple = getPointsForAnn(keyPoints, size_of_ann, verbose=False, verboseOriginalImage=originalImage)
        (pointsForAnn, topLeftPoint, bottomRightPoint) = pointsForAnnTuple
        (columns_start, rows_start) = topLeftPoint
        (columns_end, rows_end) = bottomRightPoint
        columns = columns_end - columns_start + 1
        rows = rows_end - rows_start + 1
        for degree in range(0, 360, 4) :
            rotatedPoints = rotatePoints(keyPoints, degree)
            (rotatedPointsForAnn, _, _) = getPointsForAnn(rotatedPoints, size_of_ann, verbose=False, verboseOriginalImage=originalImage)
            res = applyANN(size_of_ann, ann_net, rotatedPointsForAnn, degree, verbose=False)
            results.append((res, degree))
            if verbose:
                print imagePath," result: ", res, " degree: ",degree
        maxResult = max(results,key=lambda (result,_): result)
        (result, degree) = maxResult
        if verbose:
            print "Result: ", result, " degree: ",degree
            img2 = originalImage.copy()
            for point in keyPoints:
                cv2.circle(
                     img2,
                     (point[0].__int__(), point[1].__int__()),
                     1,
                     (0, 255, 0))
            cv2.rectangle(
                   img2,
                   topLeftPoint,
                   bottomRightPoint,
                   (0, 0, 255)
                )
            cv2.namedWindow("PointsToAnalyze", cv2.CV_WINDOW_AUTOSIZE)
            cv2.imshow("PointsToAnalyze", img2)
            cv2.waitKey(0) & 0xFF
            cv2.destroyWindow("PointsToAnalyze")
        if result >= 0.7:
            #print "accepted as line result: ", result, " degree: ",degree
            print '*',
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
                    print "Not a line, result: ", result, " degree: ",degree
                    print "Splitting points in four parts"
                    columns_mid = (columns_start + columns_end) / 2
                    rows_mid = (rows_start + rows_end) / 2
                    points_top_left = [point for point in keyPoints
                        if point[0] < columns_mid and
                            point[1] < rows_mid]
                    results_top_left = subAnalyzeImage(points_top_left, size_of_ann,
                         ann_net,
                         verbose,
                         originalImage=originalImage)
                    analyzedObjects = results_top_left + analyzedObjects
                    points_top_right = [point for point in keyPoints
                        if point[0] >= columns_mid and
                            point[1] < rows_mid]
                    results_top_right = subAnalyzeImage(points_top_right,
                        size_of_ann,
                        ann_net,
                        verbose,
                        originalImage=originalImage)
                    analyzedObjects = results_top_right + analyzedObjects
                    points_bottom_left = [point for point in keyPoints
                        if point[0] < columns_mid and
                            point[1] >= rows_mid]
                    results_bottom_left = subAnalyzeImage(points_bottom_left,
                         size_of_ann,
                         ann_net,
                         verbose,
                         originalImage=originalImage)
                    analyzedObjects = results_bottom_left + analyzedObjects
                    points_bottom_right = [point for point in keyPoints
                        if point[0] >= columns_mid and
                            point[1] >= rows_mid]
                    results_bottom_right = subAnalyzeImage(points_bottom_right,
                        size_of_ann,
                        ann_net,
                        verbose,
                        originalImage=originalImage)
                    analyzedObjects = results_bottom_right + analyzedObjects
                return analyzedObjects

def getLineObject(keyPoints):
    lineResult = (None, None)
    if len(keyPoints) > 0:
        firstPoint = keyPoints[0]
        firstX = firstPoint[0]
        firstY = firstPoint[1]
        pointA = max(keyPoints, key=lambda (x,y): abs(firstX - x) + abs(firstY - y))
        firstX = pointA[0]
        firstY = pointA[1]
        pointB = max(keyPoints, key=lambda (x,y): abs(firstX - x) + abs(firstY - y))
        lineResult = (pointA, pointB)
    return lineResult

def subAnalyseUsingGrid(keyPoints, size_of_ann, ann_net,
        verbose=False, originalImage=None):
    pointsForAnnTuple = getPointsForAnn(keyPoints, size_of_ann)
    (pointsForAnn, topLeftPoint, bottomRightPoint) = pointsForAnnTuple
    (columns_start, rows_start) = topLeftPoint
    (columns_end, rows_end) = bottomRightPoint
    columns = columns_end - columns_start + 1
    rows = rows_end - rows_start + 1
    columns_parts = columns / size_of_ann
    rows_parts = rows / size_of_ann
    result = []
    for r in range(0, rows_parts, 1):
        for c in range(0, columns_parts, 1):
            pointsToAnalyze = [point for point in keyPoints
                if point[0] >= size_of_ann * c + columns_start and
                point[0] < size_of_ann * (c + 1) + columns_start and
                point[1] >= size_of_ann * r + rows_start and
                point[1] < size_of_ann * (r + 1) + rows_start
                ]
            combinedResult = subAnalyzeImage(pointsToAnalyze,
                size_of_ann, ann_net,
                splitPointsInFourParts=False, verbose=verbose,
                originalImage=originalImage)
            result = [x for x in combinedResult
                if x[0] == "line"] + result
    return result

def analyzeImage(image, size_of_ann, ann_net, verbose = False):
    keyPoints = getKeyPoints(image, getORBKeyPoints=False, getGradientPoints = True)
    #return subAnalyzeImage(keyPoints, size_of_ann, ann_net, verbose=False, originalImage=image)
    return subAnalyseUsingGrid(keyPoints, size_of_ann, ann_net, verbose=True, originalImage=image)

def mod_dist2(pA, pB):
    return (abs(pA[0] - pB[0]) + abs(pA[1] - pB[1]))
def near(point, set, distance=1):

    return any(mod_dist2(point, p) <= distance for p in set)

def isPointAnExtension(x, y, pointsMatrix):
    return x >= 0 and y >= 0 and x < pointsMatrix.shape[1] and y < pointsMatrix.shape[0] and pointsMatrix[y][x] > 0.5 and pointsMatrix[y][x] < 1.5

def addIfExtension(bufX, bufY, pointsMatrix, accumulator):
    if isPointAnExtension(bufX, bufY, pointsMatrix):
        pointsMatrix[bufY][bufX] = 2
        accumulator = [(bufX, bufY)] + accumulator
    return accumulator

def enlargeSetByNearPoints(setToEnlarge, pointsMatrix):
    extensionSet = []
    for point in setToEnlarge:
        x = point[0]
        y = point[1]
        extensionSet = addIfExtension(x - 1, y - 1, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x, y - 1, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x + 1, y - 1, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x - 1, y, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x + 1, y, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x - 1, y + 1, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x, y + 1, pointsMatrix, extensionSet)
        extensionSet = addIfExtension(x + 1, y + 1, pointsMatrix, extensionSet)
    return extensionSet


def enlargeSetByNearPointsForSize(size, setToEnlarge,
     pointsMatrix):

    for point in setToEnlarge:
        pointsMatrix[point[1]][point[0]] = 2.0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    ((min_x, min_y), (max_x, max_y)) = getBoundingRectangle(
        setToEnlarge
        )
    accumulator = []
    while True:
        accumulator = setToEnlarge + accumulator
        ((min_x, min_y), (max_x, max_y)) = enlargeBoundingRectangle(
            setToEnlarge,
            min_x, min_y, max_x, max_y)
        if max_x - min_x >= size or max_y - min_y >= size:
            return accumulator
        newSetToEnlarge = enlargeSetByNearPoints(
            setToEnlarge,
            pointsMatrix)
        if len(newSetToEnlarge) == 0:
            return accumulator
        setToEnlarge = newSetToEnlarge


def findLines(image, size_of_ann, ann_net, verbose=False):
    function_start_time = timeit.default_timer()
    height, width, _ = image.shape

    pointsMatrix = numpyLib.zeros((height, width))
    removedPointsMatrix = numpyLib.zeros((height, width))
    keyPoints = getKeyPoints(image,
        getORBKeyPoints=False,
        getGradientPoints=True)
    for point in keyPoints:
        pointsMatrix[point[1]][point[0]] = 1.0
    result = []
    remainingPoints = keyPoints
    for pointToStartFrom in remainingPoints:
        if(removedPointsMatrix[pointToStartFrom[1]][pointToStartFrom[0]] > 0.5):
            continue
        print '.',
        sys.stdout.flush()
        #start_time = timeit.default_timer()
        pointsToAnalyze = enlargeSetByNearPointsForSize(
            size_of_ann,
            [pointToStartFrom],
            pointsMatrix)
        for point in pointsToAnalyze:
            pointsMatrix[point[1]][point[0]] = 1.0
        pointsToRemove = enlargeSetByNearPointsForSize(
            size_of_ann / 2,
            [pointToStartFrom],
            pointsMatrix)
        for point in pointsToRemove:
            removedPointsMatrix[point[1]][point[0]] = 1.0
            pointsMatrix[point[1]][point[0]] = 1.0

        combinedResult = subAnalyzeImageForLine(pointsToAnalyze,
                    size_of_ann, ann_net,
                    splitPointsInFourParts=False, verbose=verbose,
                    originalImage=image)

        result = [x for x in combinedResult
                    if x[0] == "line"] + result
        #elapsed = timeit.default_timer() - start_time
        #print "findLines iteration time: ", elapsed
    elapsed = timeit.default_timer() - function_start_time
    print "findLines total time: ", elapsed
    return result




def drawAnalyzedResults(image, results):
    cv2.destroyWindow("AnalyzedObjects")
    img2 = image.copy()
    i = 0
    for item in results:
        i+=1
        r = 0
        g = 0
        b = 0
        if i % 3 == 0:
            r = 200
        if i % 3 == 1:
            g = 200
        if i % 3 == 2:
            b = 200
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
            for point in keyPoints:
                cv2.circle(img2,
                    point,
                    1,
                    (b, g, r)
                    )
    cv2.namedWindow("AnalyzedObjects", cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("AnalyzedObjects", img2)
    cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("AnalyzedObjects")
    cv2.destroyAllWindows()

def getLinesInformation(results):
    return [getLineObject(keyPoints) for (name,
            (probability, degree),
            top_left_point,
            bottom_right_point,
            keyPoints) in results]

def drawLineInformationResults(image, results):
    cv2.destroyWindow("LinesInformation")
    img2 = image.copy()
    i = 0
    for item in results:
        i+=1
        r = 0
        g = 0
        b = 0
        if i % 3 == 0:
            r = 200
        if i % 3 == 1:
            g = 200
        if i % 3 == 2:
            b = 200
        (pointA, pointB) = item
        cv2.line(img2,
                pointA,
                pointB,
                (b, g, r), 2
                )
            
    cv2.namedWindow("LinesInformation", cv2.CV_WINDOW_AUTOSIZE)
    cv2.imshow("LinesInformation", img2)
    cv2.waitKey(0) & 0xFF
    cv2.destroyWindow("LinesInformation")
    cv2.destroyAllWindows()


def tryToCombineLines(results, size_of_ann, ann_net):
    onlyLines = [x for x in results if x[0] == "line"]
    enumeratedLines = enumerate(onlyLines)
    result = []
    for indexLeft, leftValue in enumeratedLines:
        for indexRight, rightValue in enumeratedLines:
            if indexRight != indexLeft:
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
                alpha = 20
                abs_180 = abs(180 - abs_diff)
                if abs_diff < alpha or abs_180 < alpha:
                    mergedPoints = keyPointsLeft + keyPointsRight
                    combinedResult = subAnalyzeImage(mergedPoints,
                        size_of_ann, ann_net,
                        splitPointsInFourParts=False)
                    result = [x for x in combinedResult
                        if x[0] == "line"] + result
    return result


def saveAnalyzedObjectsToFile(imagePath, analyzedObjects):
    with open(imagePath + '_analyzedObjects.txt', 'wb') as f:
        pickle.dump(analyzedObjects, f)

def drawLineAndCountSharedPoints(pointsMatrix, points):
    count = 0
    for p in points:
        pointsMatrix[p[1]][p[0]] += 1.0
        if pointsMatrix[p[1]][p[0]] > 1.1 and pointsMatrix[p[1]][p[0]] < 2.1:
            count += 1
    return count

def clearLine(pointsMatrix, points):
    for p in points:
        pointsMatrix[p[1]][p[0]] -= 1.0
        if pointsMatrix[p[1]][p[0]] < 0.1:
            pointsMatrix[p[1]][p[0]] = 0

def removeDublicatePointsFromLine(pointsMatrix, points):
    result = []
    for p in points:
        if pointsMatrix[p[1]][p[0]] > 0.1:
            result.append(p)
            pointsMatrix[p[1]][p[0]] = 0
    for p in result:
        pointsMatrix[p[1]][p[0]] = 1
    return result


def combineLines(analyzedObjects,
    analyzedLines,
    pointsMatrix,
    image,
    size_of_ann,
    ann_net):
    if len(analyzedObjects) > 0:
        first, rest = analyzedObjects[0], analyzedObjects[1:]
        drawLineAndCountSharedPoints(pointsMatrix, first[4])
        toAnalyzeFuther = []
        foundOnePair = False
        for line in rest:
            sharedPoints = drawLineAndCountSharedPoints(pointsMatrix, line[4])
            print "Shared points: ", sharedPoints
            #drawAnalyzedResults(image, [first, line])
            if sharedPoints > 10:
                resultAnalyzedOneLine = subAnalyzeImageForLine(line[4] + first[4],
                    size_of_ann, ann_net,
                    splitPointsInFourParts=False, verbose=False,
                    originalImage=image)
                resultAnalyzedOneLine = [x for x in resultAnalyzedOneLine
                    if x[0] == "line"]
                if len(resultAnalyzedOneLine) == 0:
                    print "Not merged , because pair is not a line"
                    clearLine(pointsMatrix, line[4])
                    toAnalyzeFuther.append(line)
                else:
                    print "Merged, because ",
                    print "count of lines found: ", len(resultAnalyzedOneLine)
                    (topLeft, bottomRight) = enlargeBoundingRectangle(
                        line[4],
                        first[2][0],
                        first[2][1],
                        first[3][0],
                        first[3][1]
                        )
                    combinedPoints = removeDublicatePointsFromLine(pointsMatrix,
                        line[4] + first[4])
                    first = (
                        "line",
                        first[1],
                        topLeft,
                        bottomRight,
                        combinedPoints
                        )
                    foundOnePair = True
            else:
                print "Not merged"
                clearLine(pointsMatrix, line[4])
                toAnalyzeFuther.append(line)
        clearLine(pointsMatrix, first[4])
        if foundOnePair:
            toAnalyzeFuther.append(first)
        else:
            analyzedLines.append(first)
        return combineLines(toAnalyzeFuther, analyzedLines, pointsMatrix,
            image, size_of_ann, ann_net)
    else:
        return analyzedLines

def vector(point1, point2):
    (x, y) = point1
    (a, b) = point2
    return (x - a, y - b)

def EuclidianNorm(vector):
    (x, y) = vector
    return math.sqrt(x * x + y * y)





def intersectSimple(line1, line2):
    (pointA, pointB) = line1
    (pointX, pointY) = line2
    pAX = EuclidianNorm(vector(pointA, pointX))
    pAY = EuclidianNorm(vector(pointA, pointY))
    pBX = EuclidianNorm(vector(pointB, pointX))
    pBY = EuclidianNorm(vector(pointB, pointY))
    xmin = min(pAX, pBX)
    ymin = min(pAY, pBY)
    if(ymin < xmin):
        return pointY
    else:
        return pointX

def fuzzyIsSmall(value):
    if(value < 5) :
        return True
    else: 
        return False

def fuzzyIsPointCloseToLine(line, point):
    (pointA, pointB) = line
    distA = EuclidianNorm(vector(pointA, point))
    distB = EuclidianNorm(vector(pointB, point))
    if (fuzzyIsSmall(distA) or fuzzyIsSmall(distB)):
        return True
    else:
        return False

def fuzzyIsLineSameAsVector(line, point1, point2):
    (pointA, pointB) = line
    dist1 = EuclidianNorm(vector(pointA, pointB))
    dist2 = EuclidianNorm(vector(point1, point2))
    return fuzzyIsSmall(abs(dist1 - dist2))
    

def isTriangle(line1, line2, line3):
    PL_1_2 = intersectSimple(line1, line2)
    PL_2_3 = intersectSimple(line2, line3)
    PL_1_3 = intersectSimple(line1, line3)
     
    if fuzzyIsPointCloseToLine(line1, PL_1_2) and fuzzyIsPointCloseToLine(line1, PL_1_3) and fuzzyIsPointCloseToLine(line2, PL_2_3) and fuzzyIsLineSameAsVector(line1, PL_1_2, PL_1_3) and fuzzyIsLineSameAsVector(line2, PL_2_3, PL_1_2) and fuzzyIsLineSameAsVector(line3, PL_2_3, PL_1_3):
        return True
    else:
        return False






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
    analyzedObjects = findLines(image, size_of_ann, net, verbose=False)
    drawAnalyzedResults(image, results=analyzedObjects)
    #for d in analyzedObjects:
    #    drawAnalyzedResults(image, results=[d])
    function_start_time = timeit.default_timer()
    height, width, _ = image.shape
    pointsMatrix = numpyLib.zeros((height, width))
    lines = combineLines(analyzedObjects, [], pointsMatrix, image,
        size_of_ann, net)
    for line in lines:
        print line[2], line[3], len(line[4])
    
    transformedLines = getLinesInformation(lines)
    print (transformedLines)

    elapsed = timeit.default_timer() - function_start_time
    print "time to combine lines: ", elapsed, " s"
    drawAnalyzedResults(image, results=lines)

    drawLineInformationResults(image, results=transformedLines)
    if len(transformedLines) >= 3:
        print( isTriangle(transformedLines[0], transformedLines[1], transformedLines[2]))


    #combinedObjects = tryToCombineLines(analyzedObjects, size_of_ann, net)
    #drawAnalyzedResults(image, results=combinedObjects)



