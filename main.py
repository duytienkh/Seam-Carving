import cv2 as cv
import numpy as np

def nothing(x):
    pass

def resize(h, w, img):
    img_h = len(img)
    img_w = len(img[0])

    #print(img_h, img_w)
    
    e = energy(img)

    if h < img_h:
        e = np.transpose(e, (1, 0, 2))
        img = np.transpose(img, (1, 0, 2))
        trace = findMinPath(e)
        img = remove(trace, img)
        e = np.transpose(e, (1, 0, 2))
        img = np.transpose(img, (1, 0, 2))
    if h > img_h:
        pass
    if w < img_w:
        trace = findMinPath(e)
        #img = colorPath(trace, img)
        img = remove(trace, img)
    if w > img_w:
        pass
    return img

def colorPath(trace, img, color = (0, 0, 255)):
    r, c, _ = img.shape
    for i in range(r):
        img[i][trace[i]] = color
    return img


def remove(trace, img):
    r, c, _ = img.shape
    newImg = np.zeros((r, c - 1, 3), dtype=np.uint8)

    for i in range(r):
        newImg[i] = np.delete(img[i], trace[i], axis = 0)

    return newImg

def energy(img):
    img = np.array(img, dtype=int)

    img1 = np.delete(img, 0, 0)
    img2 = np.delete(img, 0, 1)
    img1 = np.insert(img1, len(img1) - 1, [127, 127, 127], axis = 0)
    img2 = np.insert(img2, len(img2[0]) - 1, [127, 127, 127], axis = 1)

    return abs(img - img1) + abs(img - img2)

def sumRGB(pixel):
    return pixel[0] + pixel[1] + pixel[2]

def findMinPath(e):
    r, c, _ = e.shape
    dp = np.zeros((r, c), dtype=int)

    for i in range(c):
        dp[0][i] = sumRGB(e[0][i])
    
    for i in range(1, r):
        for j in range(0, c):
            minVal = dp[i - 1][j]
            if j > 0:
                minVal = min(minVal, dp[i - 1][j - 1])
            if j < c - 1:
                minVal = min(minVal, dp[i - 1][j + 1])
            dp[i][j] = minVal + sumRGB(e[i][j])

    trace = np.zeros(r, dtype=int)
    last = 0

    for i in range(1, c):
        if dp[r - 1][i] < dp[r - 1][last]:
            last = i

    trace[r - 1] = last
    for i in range(r - 1, 0, -1):
        last = trace[i]
        minVal = dp[i - 1][last]
        trace[i - 1] = last
        if last > 0:
            minVal = min(minVal, dp[i - 1][last - 1])
            if minVal == dp[i - 1][last - 1]:
                trace[i - 1] = last - 1
        if last < c - 1:
            minVal = min(minVal, dp[i - 1][last + 1])
            if minVal == dp[i - 1][last + 1]:
                trace[i - 1] = last + 1
        #trace[i] = trace[i] + i * c

    return trace


# main

root = ''
root = './Seam-Carving/'

imgPath = root + 'img1.png'
imgPath = root + 'img2.jpg'

img = cv.imread(imgPath)
cv.namedWindow('image')
cv.namedWindow('slider')
cv.createTrackbar('Height', 'slider', len(img), 2 * len(img), nothing)
cv.createTrackbar('Width', 'slider', len(img[0]), 2 * len(img[0]), nothing)

cv.imshow('slider', np.full((1, len(img[0]) + 100, 3), 255, dtype=np.uint8))

while (True):
    h = cv.getTrackbarPos('Height', 'slider')
    w = cv.getTrackbarPos('Width', 'slider')

    if h != len(img) or w != len(img[0]):
        img = resize(h, w, img)

    cv.imshow('image', img)

    #e = energy(img)
    #e = np.transpose(e, (1, 0, 2)) # = e.T

    #cv.imshow('energy', np.array(e, dtype=np.uint8))

    pressKey = cv.waitKey(1) & 0xFF

    if pressKey == ord('q'):
        break

cv.destroyAllWindows()