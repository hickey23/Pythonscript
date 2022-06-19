import cv2
import numpy as np
from PIL import Image
# image=Image.open('E:\picture\carla.png')
img_gray=cv2.imread('E:\picture\carla0.png')

# img_gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# img_gray=cv2.resize(img_gray,None,fx=0.375,fy=0.666,interpolation=cv2.INTER_LINEAR)

# cv2.imwrite('E:\picture\carla0.png',img_gray)
w,h,c=img_gray.shape
print('img_gray的shape是：',img_gray.shape)
print(w,h)
# ///////////////////////////////////////////////////////////

#三个通道修改像素值
for i in range(img_gray[:,:,0].shape[0]):
    for j in range(img_gray[:,:,1].shape[1]):
        if img_gray[:,:,0][i][j]<=42:
                img_gray[:,:,0][i][j]=255
                
for i in range(img_gray[:,:,1].shape[0]):
    for j in range(img_gray[:,:,1].shape[1]):
        if img_gray[:,:,1][i][j]<=42:
                img_gray[:,:,1][i][j]=255    
                
for i in range(img_gray[:,:,2].shape[0]):
    for j in range(img_gray[:,:,2].shape[1]):
        if img_gray[:,:,2][i][j]<=42:
                img_gray[:,:,2][i][j]=255                            
# for i in range(w):
#     for j in range(h):
#         # for k in range(c):
#             if img_gray[i,j]== [42,42,42]:
#                 img_gray[i, j] = [255,255,255]


cv2.imshow('img_gray',img_gray)
cv2.imwrite('E:\picture\carla00.png',img_gray)

# thresh,img_wb=cv2.threshold(img_gray,50,255,cv2.THRESH_BINARY_INV)
# img_wb = cv2.Canny(img_gray, 127, 255)  # 模糊图像边缘检测
thresh,img_wb=cv2.threshold(img_gray,127,255,cv2.THRESH_BINARY_INV)
cv2.imshow('img_wb',img_wb)
cv2.imwrite('E:\picture\carla000.png',img_wb)

# cv2.imshow('img',img)
cv2.waitKey(0) 