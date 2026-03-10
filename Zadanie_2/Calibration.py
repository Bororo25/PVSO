import cv2
import numpy as np
import cv2 as cv
import glob

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob('dataset5/*.png')

#cv2.namedWindow("img", cv2.WINDOW_NORMAL)
#cv2.moveWindow("img", 200, 100)
#cv2.resizeWindow("img", 900, 700)  # šírka, výška okna v pixeloch

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7, 5), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (7, 5), corners2, ret)

        cv.imshow('img', cv2.resize(img,(800,800),interpolation=cv2.INTER_AREA))
        cv.waitKey(500)

cv.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("ret:", ret)
print("camera matrix:\n", mtx)
print("dist coeffs:\n", dist)

img = cv.imread(images[0])
h, w = img.shape[:2]
newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
dst = cv.undistort(img, mtx, dist, None, newcameramtx)

x, y, w, h = roi
dst = dst[y:y+h, x:x+w]

ok = cv.imwrite("calibresult.png", dst)
print("Uložené:", ok)

mean_error = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2) / len(imgpoints2)
    mean_error += error
print("total error: {}".format(mean_error / len(objpoints)))

#show_w, show_h = 480, 480
#cv.imshow("original", cv.resize(img, (show_w, show_h)))
#cv.imshow("undistorted", cv.resize(dst, (show_w, show_h)))
#cv.waitKey(0)
#cv.destroyAllWindows()

np.savez("calibration.npz", K=mtx, dist=dist)
print("Saved to calibration.npz")