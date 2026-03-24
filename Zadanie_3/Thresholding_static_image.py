import numpy
import cv2

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

print(img1.shape)
print(img2.shape)
print(f"Prvy pixel {img1[0][0]}")

limit = 127

for i in range(img1.shape[0]):
    for j in range(img1.shape[1]):
        if(img1[i][j]>=limit):
            img1[i][j]=255
        else:
            img1[i][j]=0

for i in range(img2.shape[0]):
    for j in range(img2.shape[1]):
        if(img2[i][j]>=limit):
            img2[i][j]=255
        else:
            img2[i][j]=0

cv2.imshow("Castle",img1)
cv2.imshow("Car",img2)



cv2.waitKey(0)
cv2.destroyAllWindows()