import numpy as np
import cv2 as cv

data = np.load("calibration.npz")
K = data["K"]
dist = data["dist"]
img = cv.imread("dataset/0001.png")

h, w = img.shape[:2]
newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))  # alpha=0 = menej čiernych okrajov

und = cv.undistort(img, K, dist, None, newK)

# voliteľne orež na ROI (odstráni čierne okraje)
x, y, rw, rh = roi
und = und[y:y+rh, x:x+rw]

cv.imshow("undistorted", cv.resize(und,(800,800),interpolation=cv.INTER_AREA))
cv.waitKey(0)
cv.destroyAllWindows()