import numpy
import cv2

def histogram(img):
    hist = []
    for i in range(0,256):
        hist.append(0)

    for i in range(img.shape[0]):
        for j in range(img1.shape[1]):
            hist[img[i][j]] += 1
    return hist


img1 = cv2.imread("Castle.png")
img2 = cv2.imread("Car.png")

if img1 is None:
    print('The image 1 is empty')
if img2 is None:
    print('The image 2 is empty')


img1 = cv2.resize(img1,(1000,600),interpolation=cv2.INTER_AREA)
img2 = cv2.resize(img2,(1000,600),interpolation=cv2.INTER_AREA)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)



#cv2.imshow("Castle",img1)
cv2.imshow("Car",img2)
hist = histogram(img2)
print(hist)



cv2.waitKey(0)
cv2.destroyAllWindows()