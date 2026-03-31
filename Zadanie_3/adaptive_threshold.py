import numpy
import cv2


def adaptive_threshold_mean(img, block_size=15, C=5):
    if block_size % 2 == 0:
        block_size += 1

    output = numpy.zeros((img.shape[0], img.shape[1]), dtype=numpy.uint8)
    radius = block_size // 2

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            start_i = max(0, i - radius)
            end_i = min(img.shape[0], i + radius + 1)
            start_j = max(0, j - radius)
            end_j = min(img.shape[1], j + radius + 1)

            local_sum = 0
            count = 0

            for x in range(start_i, end_i):
                for y in range(start_j, end_j):
                    local_sum += int(img[x][y])
                    count += 1

            local_mean = local_sum / count

            if img[i][j] >= local_mean - C:
                output[i][j] = 255
            else:
                output[i][j] = 0

    return output


img1 = cv2.imread("Castle.png")
img2 = cv2.imread("Car.png")

if img1 is None:
    print('The image 1 is empty')
if img2 is None:
    print('The image 2 is empty')

img1 = cv2.resize(img1, (1000, 600), interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2, (1000, 600), interpolation=cv2.INTER_AREA)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

block_size = 15
C = 5

manual_img1 = adaptive_threshold_mean(img1, block_size, C)
manual_img2 = adaptive_threshold_mean(img2, block_size, C)

opencv_img1 = cv2.adaptiveThreshold(
    img1,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    block_size,
    C
)

opencv_img2 = cv2.adaptiveThreshold(
    img2,
    255,
    cv2.ADAPTIVE_THRESH_MEAN_C,
    cv2.THRESH_BINARY,
    block_size,
    C
)

cv2.imshow("Castle - original", img1)
cv2.imshow("Castle - adaptive threshold manual", manual_img1)
cv2.imshow("Castle - adaptive threshold OpenCV", opencv_img1)

cv2.imshow("Car - original", img2)
cv2.imshow("Car - adaptive threshold manual", manual_img2)
cv2.imshow("Car - adaptive threshold OpenCV", opencv_img2)

cv2.waitKey(0)
cv2.destroyAllWindows()