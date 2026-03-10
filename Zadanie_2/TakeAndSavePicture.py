import os
from ximea import xiapi
import cv2
import numpy as np
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

OUT_DIR = "dataset5"
os.makedirs(OUT_DIR, exist_ok=True)
counter = 0

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number)
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(200000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()

print("Ovládanie: s = ulož snímku, q = koniec ")
try:
    while True:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        preview = cv2.resize(image,(240,240),interpolation=cv2.INTER_AREA)

        cv2.imshow("test", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            counter += 1
            filename = os.path.join(OUT_DIR,f"{counter:04d}.png")
            ok = cv2.imwrite(filename,image)

            if ok:
                print(f"Uložené: {filename}")
            else:
                print("Ukladanie zlyhalo!")

# for i in range(10):
#     #get data and pass them from camera to img
#     cam.get_image(img)
#     image = img.get_image_data_numpy()
#     cv2.imshow("test", image)
#     cv2.waitKey()
#     #get raw data from camera
#     #for Python2.x function returns string
#     #for Python3.x function returns bytes
#     data_raw = img.get_image_data_raw()
#
#     #transform data to list
#     data = list(data_raw)
#
#     #print image data and metadata
#     print('Image number: ' + str(i))
#     print('Image width (pixels):  ' + str(img.width))
#     print('Image height (pixels): ' + str(img.height))
#     print('First 10 pixels: ' + str(data[:10]))
#     print('\n')

# stop data acquisition
finally:
    print('Stopping acquisition...')
    cam.stop_acquisition()
    # stop communication
    cam.close_device()
    cv2.destroyAllWindows()
    print('Done.')