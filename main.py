from ximea import xiapi
import cv2
import numpy as np
import os
from datetime import datetime

SAVE_DIR = r"C:\PVSO\Zadanie_1\frames"   # zmeň podľa seba
os.makedirs(SAVE_DIR, exist_ok=True)

def ensure_bgr(img_np):
    # ak je 4-kanál (RGB32), zober len prvé 3 kanály
    if img_np.ndim == 3 and img_np.shape[2] == 4:
        return img_np[:, :, :3].copy()
    return img_np

def rotate90_clockwise_forloops(img_np):
    # 90° doprava cez for-cykly (bez cv2.rotate/np.rot90)
    h, w, c = img_np.shape
    out = np.zeros((w, h, c), dtype=img_np.dtype)
    for y in range(h):
        for x in range(w):
            out[x, h - 1 - y] = img_np[y, x]
    return out

# create instance for first connected camera
cam = xiapi.Camera()

print('Opening first camera...')
cam.open_device()

cam.set_exposure(50000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)
print('Exposure was set to %i us' % cam.get_exposure())

img = xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

frames = []  # sem uložíme 4 snímky

print("\nOvládanie:")
print("  SPACE = zachytiť snímku (potrebuješ 4)")
print("  q     = ukonči\n")

try:
    while True:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = ensure_bgr(image)

        # (voliteľné) zmenšiť, aby bolo rovnaké vždy
        image = cv2.resize(image, (240, 240), interpolation=cv2.INTER_AREA)

        # náhľad
        preview = image.copy()
        cv2.putText(preview, f"{len(frames)}/4  SPACE=capture  q=quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("live", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        # 1) SPACE: uložiť snímku + pridať do zoznamu
        if key == 32:  # SPACE
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            path = os.path.join(SAVE_DIR, f"img_{len(frames)+1}_{ts}.png")
            cv2.imwrite(path, image)
            frames.append(image)
            print("Saved:", path)

            # keď máme 4, sprav mozaiku + úpravy
            if len(frames) == 4:
                h, w = frames[0].shape[:2]

                # 2) mozaika 2x2 indexovaním
                mosaic = np.zeros((2*h, 2*w, 3), dtype=frames[0].dtype)
                mosaic[0:h,   0:w]   = frames[0]  # časť 1
                mosaic[0:h,   w:2*w] = frames[1]  # časť 2
                mosaic[h:2*h, 0:w]   = frames[2]  # časť 3
                mosaic[h:2*h, w:2*w] = frames[3]  # časť 4

                # 6) výpis info
                print("\n--- Mosaic info ---")
                print("dtype:", mosaic.dtype)
                print("shape:", mosaic.shape)
                print("size:", mosaic.size)
                print("-------------------\n")

                # 3) časť 1: filter2D 3x3 (napr. doostrenie)
                kernel = np.array([[0, -1, 0],
                                   [-1, 5, -1],
                                   [0, -1, 0]], dtype=np.float32)
                part1 = mosaic[0:h, 0:w]
                mosaic[0:h, 0:w] = cv2.filter2D(part1, -1, kernel)

                # 4) časť 2: rotácia 90° cez for-cykly
                part2 = mosaic[0:h, w:2*w]
                rot = rotate90_clockwise_forloops(part2)
                # rot má tvar (w,h,3) – keďže máme 240x240, sedí to.
                mosaic[0:h, w:2*w] = rot

                # 5) časť 3: iba červený kanál (OpenCV = BGR, červený je index 2)
                part3 = mosaic[h:2*h, 0:w]
                red_only = np.zeros_like(part3)
                red_only[:, :, 2] = part3[:, :, 2]
                mosaic[h:2*h, 0:w] = red_only

                cv2.imshow("mosaic_processed", mosaic)

                # čakaj, kým stlačíš q (v okne)
                while True:
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        raise SystemExit

finally:
    print('Stopping acquisition...')
    try:
        cam.stop_acquisition()
    except Exception:
        pass
    try:
        cam.close_device()
    except Exception:
        pass
    cv2.destroyAllWindows()
    print('Done.')
