# from turtle import width
# import cv2
# import numpy as np
# def otsu(gray):
#     pixel_number = gray.shape[0] * gray.shape[1]
#     mean_weigth = 1.0/pixel_number
#     his, bins = np.histogram(gray, np.arange(0,257))
#     final_thresh = -1
#     final_value = -1
#     intensity_arr = np.arange(256)
#     for t in bins[1:-1]: 
#         pcb = np.sum(his[:t])

#         pcf = np.sum(his[t:])
#         Wb = pcb * mean_weigth
#         Wf = pcf * mean_weigth
#         mub = np.sum(intensity_arr[:t]*his[:t]) / float(pcb)
#         muf = np.sum(intensity_arr[t:]*his[t:]) / float(pcf)
#         #print mub, muf
#         value = Wb * Wf * (mub - muf) ** 2
#         if value > final_value:
#             final_thresh = t
#             final_value = value
#     final_img = gray.copy()
#     print(final_thresh)
#     final_img[gray > final_thresh] = 255
#     final_img[gray < final_thresh] = 0
#     return final_img
# img=cv2.imread('E:\picture\Lenna.png')
# gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
# # cv2.imshow('1',img)
# otsu=otsu(gray=gray)
# cv2.imshow('otsu',otsu)
# cv2.imshow('2',gray)
# cv2.waitKey(0)

# s=img.shape
# for i in range(s[0]):
#     for j in range(s[1]):
#         print(img[i,j])        
#         if img[i,j]==255:
#             img[i,j]=0
#         else:
#             img[i,j]=img[i,j]
# cv2.imshow('img',img)
# cv2.waitKey(0)               

# RGB2HSI
import numpy as np
# from PIL import Image
import math
import copy
import matplotlib.pyplot as pl
import cv2
import numpy as np
def RGB2HSI(img1):
    img1 = img1.astype('float32')
    b, g, r = img1[:, :, 0]/255.0, img1[:, :, 1]/255.0, img1[:, :, 2]/255.0

    I = (r+g+b)/3.0

    tem = np.where(b >= g, g, b)
    minValue = np.where(tem >= r, r, tem)
    S = 1 - (3 / (r + g + b)) * minValue

    num1 = 2*r - g - b
    num2 = 2*np.sqrt(((r - g) ** 2) + (r-b)*(g-b))
    deg = np.arccos(num1/num2)
    H = np.where(g >= b, deg, 2*np.pi - deg)

    resImg = np.zeros((img1.shape[0], img1.shape[1],
                    img1.shape[2]), dtype=np.float)
    resImg[:, :, 0], resImg[:, :, 1], resImg[:, :, 2] = H*255, S*255, I*255
    resImg = resImg.astype('uint8')
    return resImg


def HSI2RGB(img):
    H1, S1, I1 = img[:,:,0]/255.0, img[:,:,1]/255.0, img[:,:,2]/255.0
    B = np.zeros((H1.shape[0], H1.shape[1]), dtype='float32')
    G = np.zeros((S1.shape[0], S1.shape[1]), dtype='float32')
    R = np.zeros((I1.shape[0], I1.shape[1]), dtype='float32')
    H = np.zeros((H1.shape[0], H1.shape[1]), dtype='float32')
    
    for i in range(H1.shape[0]):
        for j in range(H1.shape[1]):
            H = H1[i][j]
            S = S1[i][j]
            I = I1[i][j]   
            if (H >=0) & (H < (np.pi * (2/3))):
                B[i][j] = I*(1-S)
                R[i][j] = I * (1 + ((S*np.cos(H))/np.cos(np.pi * (1/3) - H)))
                G[i][j] = 3*I - (B[i][j]+R[i][j])
                
            elif (H >= (np.pi * (2/3))) & (H < np.pi * (4/3)):
                R[i][j] = I*(1-S)
                G[i][j] = I * (1 + ((S*np.cos(H - np.pi * (2/3)))/np.cos(np.pi * (1/2) - H)))
                B[i][j] = 3*I - (G[i][j]+R[i][j])
            elif (H >= (np.pi * (2/3))) & (H < (np.pi * 2)):
                G[i][j] = I*(1-S)
                B[i][j] = I * (1 + ((S*np.cos(H - np.pi * (4/3)))/np.cos(np.pi * (10/9) - H)))
                R[i][j] = 3*I - (G[i][j]+B[i][j])
    img = cv2.merge((B*255, G*255, R*255))
    img = img.astype('uint8')
    return img



img =cv2.imread('E:\picture\Lenna.png')
# img=cv2.imread('E:\picture\Lenna.png')
imghsi=RGB2HSI(img)
# cv2.imshow('1',img)
# cv2.waitKey(0)
cv2.imshow('1',imghsi)
cv2.waitKey(0)
# img = np.array(Image.open('t9.jpg',).convert('RGB'))    # 打开图片转换为numpy数组
new_r = copy.deepcopy(imghsi)                              # 进行拷贝，为后面的分量显示做铺垫
new_g = copy.deepcopy(imghsi)
new_b = copy.deepcopy(imghsi)
imghsi = imghsi/255                                           # 归一化

r = imghsi[:, :, 0]
rows = len(r)
cols = len(r[1])
new = copy.deepcopy(imghsi)                                # 用来储存HSI图像

for i in range(rows):
    for j in range(cols):
        II = (imghsi[i, j, 0]+imghsi[i, j, 1]+imghsi[i, j, 2])/3 #HSI分量的I

        num = 0.5 * ((imghsi[i, j, 0]-imghsi[i, j, 1]) + (imghsi[i, j, 0]-imghsi[i, j, 2]))     # theta值的分子

        den = math.sqrt((imghsi[i, j, 0]-imghsi[i, j, 1]) ** 2 + (imghsi[i, j, 0]-imghsi[i, j, 2])*(imghsi[i, j, 1]-imghsi[i, j, 2]))     # theta值的分母
        if den == 0:    # 分母不为0
            theta = 0
        else:
            theta = math.acos(num/den)  # 求theta
        if img[i, j, 2] <= img[i, j, 1]:    # B<=G
            new[i, j, 0] = int(theta*255/(2*math.pi))
        else:
            new[i, j, 0] = int((2*math.pi-theta)*255/(2*math.pi))
        if II == 0:         # I为0时设置S为0
            new[i, j, 1] = 0
        else:
            new[i, j, 1] = int((1-min((imghsi[i, j, 0], imghsi[i, j, 1], imghsi[i, j, 2]))/II)*255)
        new[i, j, 2] = int(II*255)
# '''上面×255的操作也可以完成后再进行'''

imge = np.array(new)    # 把new转化成numpy数组

im = imge.astype(np.int)    # numpy数组转化为0-255的整数

new_r[:, :, 1] = 0
new_r[:, :, 2] = 0
RGB_R = new_r   # R分量显示，把GB设为0

new_g[:, :, 0] = 0
new_g[:, :, 2] = 0
RGB_G = new_g   # G分量显示，把RB设为0

new_b[:, :, 0] = 0
new_b[:, :, 1] = 0
RGB_B = new_b   # B分量显示，把RG设为0


pl.subplot(141)     # 一行四列第一个图
pl.imshow(im)
pl.title('HSI')
pl.axis('off')      # 不显示坐标


pl.subplot(142)
pl.imshow(RGB_R)
pl.title('R')
pl.axis('off')


pl.subplot(143)
pl.imshow(RGB_G)
pl.title('G')
pl.axis('off')

pl.subplot(144)
pl.imshow(RGB_B)
pl.title('B')
pl.axis('off')

# pl.savefig("out.jpg")   
pl.show()              

 











