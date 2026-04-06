import numpy
import cv2
import time

# def histogram(img):
#     hist = []
#     for i in range(0,255):
#         hist.append(0)
#
#     for i in range(img.shape[0]):
#         for j in range(img1.shape[1]):
#             #hist[img[i][j]] += 1
#             #print(img[i][j])
#     return hist


img = cv2.imread("Castle.png")
# img2 = cv2.imread("Car.png")

if img is None:
    print('The image 1 is empty')
# if img2 is None:
#     print('The image 2 is empty')


img = cv2.resize(img,(1000,600),interpolation=cv2.INTER_AREA)
# img2 = cv2.resize(img2,(1000,600),interpolation=cv2.INTER_AREA)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

print(img.shape)
# print(img2.shape)
#print(f"Prvy pixel {img1[0][0]}")


# limit = 100
for limit in range(20,230,10):
    img1 = img.copy()
    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            if(img1[i][j]>=limit):
                img1[i][j]=255
            else:
                img1[i][j]=0



    cv2.imshow("Castle",img1)
    cv2.imwrite(f"Car-basic-treshold-t={limit}.png",img1)
    cv2.waitKey(500)
# cv2.imshow("Car",img2)
#hist = histogram(img2)
#print(hist)



# cv2.waitKey(0)
# cv2.destroyAllWindows()