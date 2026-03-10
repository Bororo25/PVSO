import numpy as np
import cv2 as cv

# kalibrácia
data = np.load("calibration.npz")
camera_matrix = data["K"]
dist_coeffs = data["dist"]

cap = cv.VideoCapture(0)

def classify_shape(contour):
    peri = cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, 0.04 * peri, True)
    area = cv.contourArea(contour)

    if area < 500:
        return None, approx

    if len(approx) == 3:
        return "Trojuholnik", approx

    elif len(approx) == 4:
        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 0.95 <= aspect_ratio <= 1.05:
            return "Stvorec", approx
        else:
            return "Obdlznik", approx

    return None, approx

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # odstranenie skreslenia
    frame = cv.undistort(frame, camera_matrix, dist_coeffs)

    output = frame.copy()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    blur = cv.medianBlur(gray, 5)

    # 1. kruznice
    circles = cv.HoughCircles(
        blur, cv.HOUGH_GRADIENT, 1, 20,
        param1=50, param2=30, minRadius=0, maxRadius=0
    )

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(output, (i[0], i[1]), i[2], (0, 255, 0), 2)
            cv.circle(output, (i[0], i[1]), 2, (0, 0, 255), 3)
            cv.putText(output, "Kruznica", (i[0] - 40, i[1] - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # 2. ostatne tvary
    edges = cv.Canny(blur, 50, 150)
    contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        shape_name, approx = classify_shape(cnt)
        if shape_name is None:
            continue

        cv.drawContours(output, [approx], -1, (0, 255, 255), 2)

        M = cv.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv.circle(output, (cx, cy), 4, (0, 0, 255), -1)
            cv.putText(output, shape_name, (cx - 40, cy - 10),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    cv.imshow("Detekcia tvarov", output)
    cv.imshow("Hrany", edges)

    key = cv.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
cv.destroyAllWindows()