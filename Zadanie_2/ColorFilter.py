import cv2 as cv
import numpy as np

# -----------------------------
# 1. Nacitanie kalibracie
# -----------------------------
calib = np.load("calibration.npz")
cameraMatrix = calib["K"]
distCoeffs = calib["dist"]

# -----------------------------
# 2. Kamera
# -----------------------------
cap = cv.VideoCapture(0)

if not cap.isOpened():
    print("Kamera sa nepodarila otvorit.")
    exit()

# Nastavenie rozlisenia pre plynulejsi real-time beh
cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

# Zistime realnu velkost frame pre optimalizaciu undistortion
ret, frame = cap.read()
if not ret:
    print("Nepodarilo sa nacitat prvy frame.")
    cap.release()
    exit()

h, w = frame.shape[:2]

# Nova optimalna kamera
newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(
    cameraMatrix, distCoeffs, (w, h), 1, (w, h)
)

# -----------------------------
# 3. Spracovanie v cykle
# -----------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Nepodarilo sa nacitat frame.")
        break

    # ---------------------------------
    # A) Odstranenie skreslenia obrazu
    # ---------------------------------
    undistorted = cv.undistort(frame, cameraMatrix, distCoeffs, None, newCameraMatrix)

    # Ak chces, mozes obraz orezat na validnu oblast
    x, y, w_roi, h_roi = roi
    if w_roi > 0 and h_roi > 0:
        undistorted_cropped = undistorted[y:y+h_roi, x:x+w_roi]
    else:
        undistorted_cropped = undistorted

    # ---------------------------------
    # B) Farebny filter v HSV
    #    priklad: cervena -> zelena
    # ---------------------------------
    hsv = cv.cvtColor(undistorted_cropped, cv.COLOR_BGR2HSV)

    # Cervena je v HSV rozdelena na 2 intervaly
    lower_red1 = np.array([0, 100, 80])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 80])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv.inRange(hsv, lower_red2, upper_red2)
    mask = cv.bitwise_or(mask1, mask2)

    # Vyhladenie/ocistenie masky
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Nahradenie cervenej farby zelenou
    output = undistorted_cropped.copy()
    output[mask > 0] = [0, 255, 0]   # BGR zelena

    # ---------------------------------
    # C) Zobrazenie
    # ---------------------------------
    cv.imshow("Original", frame)
    cv.imshow("Undistorted", undistorted_cropped)
    cv.imshow("Mask", mask)
    cv.imshow("Color Filter", output)

    key = cv.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break

cap.release()
cv.destroyAllWindows()