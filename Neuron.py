import numpy as numpyLib
import neurolab as neurolabLib
import cv2
import math
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt



def getPointsFromImage(fileName, size_of_ann):
    #img = cv2.imread('messi5.jpg',0)
    img = cv2.imread(fileName, 0)
    # Initiate STAR detector
    orb = cv2.ORB(1800)

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    #kp, des = orb.compute(img, kp)
    mid_x = 0
    mid_y = 0
    min_x = 0
    min_y = 0
    max_x = 0
    max_y = 0
    if len(kp) > 0:
        mid_x = 0
        mid_y = 0
        min_x = kp[0].pt[0]
        min_y = kp[0].pt[1]
        max_x = min_x
        max_y = min_y
        for point in kp:
            x = point.pt[0]
            y = point.pt[1]
            mid_x += x
            mid_y += y
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
        mid_x /= len(kp)
        mid_y /= len(kp)
        print mid_x, ":",  mid_y
    max_dist_from_center = max(mid_x - min_x, max_x - mid_x, mid_y - min_y, max_y - mid_y)
    ratio = size_of_ann / (2 * max_dist_from_center)
    min_x_i = math.floor(min_x).__int__()
    min_y_i = math.floor(min_y).__int__()
    max_x_i = math.floor(max_x).__int__()
    max_y_i = math.floor(max_y).__int__()
    print min_x, min_y, max_x, max_y
    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0), flags=0)
    cv2.line(img2, (min_x_i, min_y_i), (min_x_i, max_y_i), (0, 255, 0), 4)
    cv2.line(img2, (min_x_i, min_y_i), (max_x_i, min_y_i), (0, 255, 0), 4)
    cv2.line(img2, (max_x_i, max_y_i), (max_x_i, min_y_i), (0, 255, 0), 4)
    cv2.line(img2, (min_x_i, max_y_i), (max_x_i, max_y_i), (0, 255, 0), 4)
    plt.imshow(img2), plt.show()

    result_vector = numpyLib.zeros((size_of_ann * size_of_ann, 1))

    for point in kp:
        x = point.pt[0]
        y = point.pt[1]
        x -= mid_x
        y -= mid_y
        x *= ratio
        y *= ratio
        x += size_of_ann / 2
        y += size_of_ann / 2
        cv2.circle(img2, (x.__int__(), y.__int__()), 1, (0, 255, 0))
        index = math.floor(y * size_of_ann + x).__int__()
        index = min(index, size_of_ann * size_of_ann)
        index = max(index, 0)
        result_vector[index] = 1
    plt.imshow(img2), plt.show()
    return result_vector


imagesNames = [f for f in listdir("./") if isfile(join("./", f)) and
    (f.endswith(".png") or f.endswith(".jpg"))]
learningImages = [f for f in imagesNames if f.startswith("success_") or f.startswith("failure_")]

print imagesNames

size_of_ann = 10
count_of_images = len(learningImages)
print count_of_images
learning_set = numpyLib.zeros((count_of_images, size_of_ann * size_of_ann))
target_set = numpyLib.zeros((count_of_images, 1))
for imageIndex in range(len(learningImages)):
    imagePath = learningImages[imageIndex]
    cur_result = getPointsFromImage(imagePath, size_of_ann)
    to_set = cur_result.reshape((size_of_ann * size_of_ann))
    learning_set[imageIndex, :] = to_set
    if imagePath.startswith("success_"):
        target_set[imageIndex, 0] = 1
    elif imagePath.startswith("failure_"):
        target_set[imageIndex, 0] = 0
boundaries = numpyLib.zeros((size_of_ann * size_of_ann, 2))
boundaries[:, 1] = numpyLib.ones((size_of_ann * size_of_ann, 1)).reshape((size_of_ann * size_of_ann))


#cv2.imshow('image',img)
#k = cv2.waitKey(0) & 0xFF
#if k == 27:         # wait for ESC key to exit
#    cv2.destroyAllWindows()

# Create train samples
# train sum function

# Create network with 2 inputs, 5 neurons in input layer
# And 1 in output layer
net = neurolabLib.net.newff(boundaries, [14, 1])
# Train process
err = net.train(learning_set, target_set, show=2, epochs=10, goal=0.0001)
# Test
for imagePath in imagesNames:
    res = net.sim(getPointsFromImage(imagePath, size_of_ann).reshape(1, size_of_ann * size_of_ann))
    print imagePath, res

# Plot result
import pylab as pl
pl.subplot(211)
pl.plot(err)
pl.xlabel('Epoch number')
pl.ylabel('error (default SSE)')
pl.show()
