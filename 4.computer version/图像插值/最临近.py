import cv2
import numpy as np


def function(img):
    height, width, channels = img.shape
    emptyImage = np.zeros((960, 960, channels), np.uint8)
    sh = 960 / height
    sw = 960 / width
    for i in range(960):
        for j in range(960):
            x = int(i / sh)
            y = int(j / sw)
            emptyImage[i, j] = img[x, y]
    return emptyImage


img = cv2.imread("D:\\GitHub\\tensorflow_from_my_notebook\\4.computer version\\images\\test-INSERT.jpg")
zoom = function(img)
cv2.imshow("nearest neighbor", zoom)
cv2.imshow("image", img)
cv2.waitKey(0)
