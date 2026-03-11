import numpy as np
import cv2 as cv

def nothing(x):
    pass

# Create a black image, a window
img = np.zeros((300,512,3), np.uint8)
cv.namedWindow('image')

# create trackbars for color change
cv.createTrackbar('H','image',0,255,nothing)

cv.createTrackbar('S','image',0,255,nothing)
cv.createTrackbar('V','image',0,255,nothing)

# create switch for ON/OFF functionality
switch = '0 : OFF \n1 : ON'
cv.createTrackbar(switch, 'image',0,1,nothing)

while(1):
    cv.imshow('image',img)
    k = cv.waitKey(1) & 0xFF
    if k == 27:
        break

    # get current positions of four trackbars
    h = cv.getTrackbarPos('H','image')
    s = cv.getTrackbarPos('S','image')
    v = cv.getTrackbarPos('V','image')
    sa = cv.getTrackbarPos(switch,'image')

    pixel_hsv = np.array([[[h, s, v]]], dtype=np.uint8)
    pixel_bgr = cv.cvtColor(pixel_hsv, cv.COLOR_HSV2BGR)

    #print(pixel_bgr)
    b, g, r = pixel_bgr[0, 0]

    if sa == 0:
        img[:] = 0
    else:
        img[:] = [b,g,r]

cv.destroyAllWindows()