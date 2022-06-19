import os
import cv2
for i in range(181):
    print(i)
    img = cv2.imread("E:\picture\pic1\carlapic\{}.jpg".format(i))
    print("img的形状:",img.shape) # (1080, 1920, 3)
    cropped = img[240:720, 400:880]  # 裁剪坐标为[y0:y1, x0:x1]
    # cv2.imwrite("./data/cut/cv_cut_thor.jpg", cropped)
    print('裁剪后的图片形状：',cropped.shape)

    cropped_gray=cv2.cvtColor(cropped,cv2.COLOR_RGB2GRAY)
    w,h=cropped_gray.shape
    for a in range(w):
        for j in range(h):
            if cropped_gray[a,j]<=42:
                cropped_gray[a,j]=255
                    
    thresh,dst=cv2.threshold(cropped_gray,90,255,cv2.THRESH_BINARY_INV)
    print(dst.shape)
    print(dst)
    cv2.imwrite('E:\picture\pic1\heibai\heibai{}.jpg'.format(i),dst)
    # cv2.imshow('cropped',cropped)
    # cv2.imshow('1',cropped_gray)
    # cv2.imshow('2',dst)
    cv2.waitKey(0)
    print('/////////////////////////////////////////////////////')
    print('/////////////////////////////////////////////////////')
    print('finish')