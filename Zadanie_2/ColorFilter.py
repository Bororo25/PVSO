import os
from operator import truediv

import numpy as np
from ximea import xiapi
import cv2 as cv

data = np.load("calibration.npz")
K = data["K"]
dist = data["dist"]

cam = xiapi.Camera()

print('Opening first camera...')
cam.open_device()
cam.set_exposure(100000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())
img = xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

try:
    while True:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        preview = cv.resize(image,(500,500),interpolation=cv.INTER_AREA)
        h, w = preview.shape[:2]
        newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), 0, (w, h))  # alpha=0 = menej čiernych okrajov
        preview = cv.undistort(preview, K, dist, None, newK)

        #print(preview.shape)

        hsv = cv.cvtColor(preview, cv.COLOR_BGR2HSV)


        lower = np.array([0, 100, 75], dtype=np.uint8)
        upper = np.array([7, 255, 255], dtype=np.uint8)

        mask = cv.inRange(hsv, lower, upper)


        img_filtered = preview[:, :, :3].copy()
        img_filtered[mask == 255] = (0, 200, 255)
        cv.imshow("img_filtered", img_filtered)

        key = cv.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            filename = os.path.basename("filtrovany2.png")
            ok1 = cv.imwrite(filename, img_filtered)
            filename = os.path.basename("povodny2.png")
            ok2 = cv.imwrite(filename, preview)

            if ok1 and ok2:
                print(f"Uložené: {filename}")
            else:
                print("Ukladanie zlyhalo!")
finally:
    print('Stopping acquisition...')
    cam.stop_acquisition()
    # stop communication
    cam.close_device()
    cv.destroyAllWindows()
    print('Done.')



