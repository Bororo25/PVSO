from operator import truediv

import numpy as np
import cv2 as cv
import time

img = cv.imread("calibration_b.png")
if img is None:
    print('The image is empty')


print(img.shape)
#print(img.dtype)
#print(img[50,50,:]) #cerveny pixel

hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
#print(hsv.shape)
#print(hsv[50,50,:]) #cerveny pixel v hsv

lower = np.array([0, 150, 75], dtype=np.uint8)
upper = np.array([10, 255, 255], dtype=np.uint8)

mask = cv.inRange(hsv, lower, upper)
#cv.imshow("mask1", mask1)
#redPart = cv.bitwise_and(img,img,mask=mask1)
#cv.imshow('redPart',redPart)
#print(mask1.shape)

img_filtered = img
img_filtered[mask == 255] = (0, 200, 255)
cv.imshow("img_filtered", img_filtered)
cv.waitKey(0)
cv.destroyAllWindows()
