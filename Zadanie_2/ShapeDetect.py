from ximea import xiapi
import cv2
import numpy as np

# Nastavenia

EXPOSURE_US = 100000
ROI_RATIO = 1
SAT_THRESHOLD = 40
MIN_AREA = 8000
BORDER_MARGIN = 12
APPROX_EPS = 0.04


# Pomocné funkcie
def touches_border(contour, width, height, margin=BORDER_MARGIN):
    x, y, w, h = cv2.boundingRect(contour)

    if x <= margin:
        return True
    if y <= margin:
        return True
    if x + w >= width - margin:
        return True
    if y + h >= height - margin:
        return True

    return False


def contour_center(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0:
        return None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    return cx, cy


def build_color_mask(frame, sat_threshold):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1]

    s_blur = cv2.GaussianBlur(s, (7, 7), 0)

    _, mask = cv2.threshold(s_blur, sat_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def classify_and_draw_shape(cnt, output):
    area = cv2.contourArea(cnt)
    if area < MIN_AREA:
        return
    #obvod
    perimeter = cv2.arcLength(cnt, True)
    if perimeter == 0:
        return

    hull = cv2.convexHull(cnt)
    approx = cv2.approxPolyDP(hull, APPROX_EPS * perimeter, True)

    x, y, w, h = cv2.boundingRect(hull)
    if h == 0:
        return
    #pomer
    aspect_ratio = w / float(h)
    circularity = 4 * np.pi * area / (perimeter * perimeter + 1e-6)

    center = contour_center(cnt)
    if center is None:
        return

    cx, cy = center


    # 1. Trojuholnik

    if len(approx) == 3:
        cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
        cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)
        cv2.putText(
            output,
            "trojuholnik",
            (cx - 60, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )
        return


    # 2. Stvorec / Obdlznik
    if len(approx) == 4:
        rect = cv2.minAreaRect(hull)
        rw, rh = rect[1]

        if rw == 0 or rh == 0:
            return

        rect_aspect = max(rw, rh) / float(min(rw, rh))

        cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
        cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

        if rect_aspect <= 1.15:
            label = "stvorec"
        else:
            label = "obdlznik"

        cv2.putText(
            output,
            label,
            (cx - 50, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )
        return

    # 3. Kruh
    if len(approx) > 4 and circularity > 0.78 and 0.75 <= aspect_ratio <= 1.25 and len(cnt) >= 5:
        ellipse = cv2.fitEllipse(cnt)

        cv2.ellipse(output, ellipse, (0, 255, 0), 3)
        cv2.circle(output, (cx, cy), 6, (0, 0, 255), -1)

        cv2.putText(
            output,
            "kruh",
            (cx - 30, cy - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 0, 0),
            2
        )
        return


def detect_shapes(frame, sat_threshold=40):
    output = frame.copy()
    height, width = frame.shape[:2]

    mask = build_color_mask(frame, sat_threshold)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if touches_border(cnt, width, height):
            continue

        classify_and_draw_shape(cnt, output)

    return output, mask


# -------------------------------------------------
# XIMEA kamera
# -------------------------------------------------

cam = xiapi.Camera()
img = xiapi.Image()

print("Opening first camera...")
cam.open_device()

cam.set_exposure(EXPOSURE_US)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

print("Exposure was set to %i us" % cam.get_exposure())

print("Starting data acquisition...")
cam.start_acquisition()

print("Ovládanie:")
print("  q = koniec")
print("  m = zap/vyp maska")
print("  + = zvys SAT threshold")
print("  - = zniz SAT threshold")

show_mask = True
sat_threshold = SAT_THRESHOLD
exposure_us = EXPOSURE_US
try:
    while True:
        cam.get_image(img)
        image = img.get_image_data_numpy()

        if len(image.shape) == 3 and image.shape[2] == 4:
            frame = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        else:
            frame = image.copy()

        h, w = frame.shape[:2]

        roi_w = int(w * ROI_RATIO)
        roi = frame[:, :roi_w]

        detected_roi, mask = detect_shapes(roi, sat_threshold=sat_threshold)

        output = frame.copy()
        output[:, :roi_w] = detected_roi

        cv2.putText(
            output,
            f"SAT threshold: {sat_threshold}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        cv2.putText(
            output,
            f"Exposure: {exposure_us} us",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 255),
            2
        )
        preview = cv2.resize(output, (1000, 700), interpolation=cv2.INTER_AREA)
        cv2.imshow("Live detekcia tvarov", preview)

        if show_mask:
            mask_preview = cv2.resize(mask, (600, 450), interpolation=cv2.INTER_AREA)
            cv2.imshow("Maska", mask_preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break
        elif key == ord("m"):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow("Maska")
        elif key in (ord('+'), ord('=')):
            sat_threshold = min(255, sat_threshold + 2)
            print(f"SAT threshold: {sat_threshold}")
        elif key == ord('-'):
            sat_threshold = max(0, sat_threshold - 2)
            print(f"SAT threshold: {sat_threshold}")
        elif key == ord('i'):
            exposure_us = min(1000000, exposure_us + 5000)
            cam.set_exposure(exposure_us)
            print(f"Exposure: {exposure_us} us")

        elif key == ord('k'):
            exposure_us = max(1000, exposure_us - 5000)
            cam.set_exposure(exposure_us)
            print(f"Exposure: {exposure_us} us")


finally:
    print("Stopping acquisition...")
    cam.stop_acquisition()
    cam.close_device()
    cv2.destroyAllWindows()
    print("Done.")