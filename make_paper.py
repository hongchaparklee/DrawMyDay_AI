import cv2
import numpy as np

img = np.ndarray((600,800,3))


xnum = int(800/10)
ynum = int(600/10)

ind1 = int(800/64)
ind2 = int(600/64)
print(ind1)
print(ind2)

img = 255-img
print(img.shape)

for i in range(ind1+1) :
    for j in range(600) :
        img[j][i*64][0] = 0
        img[j][i*64][1] = 0

for i in range(ind2+1) :
    for j in range(800) :
        img[i*64][j][0] = 0
        img[i*64][j][1] = 0


for i in range(12) :
    for j in range(9) :
        tmp = img[j*64+1:j*64+64,i*64+1 : i*64+64]
        print(tmp.shape)
        cv2.imshow('tmp',tmp)
        cv2.waitKey(0)
    
cv2.imshow('img',img)
cv2.imwrite('paper.png',img)

# 사용자가 키보드의 아무 키나 누를 때까지 대기합니다.
cv2.waitKey(0)

# 모든 윈도우를 닫습니다.
cv2.destroyAllWindows()