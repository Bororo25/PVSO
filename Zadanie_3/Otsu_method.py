import numpy as np
import cv2
import matplotlib.pyplot as plt


def histogram(img):
    hist = [0] * 256

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            hist[img[i, j]] += 1

    return hist


def otsu_threshold(img):
    hist = histogram(img)
    total = img.shape[0] * img.shape[1]

    best_t = 0
    max_sigma = 0
    sigma_values = []

    for t in range(256):

        omega0 = 0
        omega1 = 0

        for i in range(t+1):
            omega0 += hist[i] / total
        for i in range(t+1, 256):
            omega1 += hist[i] / total

        if omega0 == 0 or omega1 == 0:
            sigma_values.append(0)
            continue

        mi0_upper = 0
        mi1_upper = 0

        for i in range(t+1):
            mi0_upper += i * hist[i] / total
        mi0 = mi0_upper / omega0

        for i in range(t+1,256):
            mi1_upper += i * hist[i] / total
        mi1 = mi1_upper / omega1

        sigma2 = omega0 * omega1 * (mi0 - mi1) ** 2
        sigma_values.append(sigma2)

        if sigma2 > max_sigma:
            max_sigma = sigma2
            best_t = t

    return best_t, sigma_values


def apply_threshold(img, threshold):
    out = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > threshold:
                out[i, j] = 255
            else:
                out[i, j] = 0

    return out


img1 = cv2.imread("Car.png")

if img1 is None:
    print("The image 1 is empty")

img1 = cv2.resize(img1, (1000, 600), interpolation=cv2.INTER_AREA)

img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# histogram
# hist = histogram(img1)
# print("Histogram:")
# print(hist)

# Otsu threshold
t, sigma_values = otsu_threshold(img1)
print("Otsu threshold:", t)

# binárny obraz
binary = apply_threshold(img1, t)

cv2.imshow("Original", img1)
cv2.imwrite("Otsu-Car-gray.png", img1)
cv2.imshow("Otsu Binary", binary)
cv2.imwrite("Otsu-Car-treshold.png", binary)

t1, binary = cv2.threshold(img1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu threshold OpenCV:", t1)
cv2.imshow("Otsu Binary OpenCV", binary)
cv2.imwrite("Otsu-Car-threshold-OpenCV.png", binary)


# hist = cv2.calcHist([img1], [0], None, [256], [0, 256])
#
# plt.bar(range(256), hist.flatten())
# plt.axvline(x=t, color="red", linestyle="--", label=f"Otsu threshold = {int(t)}")
# plt.title("Histogram obrazu")
# plt.xlabel("Intenzita jasu")
# plt.ylabel("Pocet pixelov")
# plt.xlim([0, 256])
# plt.savefig("histogram_Car.png", dpi=300, bbox_inches="tight")
# plt.show()

hist = histogram(img1)

fig, ax1 = plt.subplots(figsize=(12, 6))

# histogram
ax1.bar(range(256), hist, color="lightgray", alpha=0.8, label="Histogram")
ax1.set_xlabel("Intenzita jasu / prah t")
ax1.set_ylabel("Počet pixelov")
ax1.set_xlim([0, 255])

# zvislá čiara pre Otsu threshold
ax1.axvline(x=t, color="red", linestyle="--", linewidth=2, label=f"Otsu threshold = {t}")

# druhá os pre sigma²
ax2 = ax1.twinx()
ax2.plot(range(256), sigma_values, color="blue", linewidth=2, label="sigma²(t)")
ax2.set_ylabel("sigma²")

# zvýraznenie maxima sigma²
ax2.plot(t, sigma_values[t], "ro", label=f"maximum sigma² pri t = {t}")

# spoločná legenda
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

plt.title("Histogram obrazu a priebeh sigma² pre Otsuho metódu")
plt.savefig("histogram_Car_sigma2_one_plot.png", dpi=300, bbox_inches="tight")
plt.show()

# combined = np.vstack((img1, binary))
# cv2.imshow("Combined", combined)

cv2.waitKey(0)
cv2.destroyAllWindows()