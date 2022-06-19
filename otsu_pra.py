import math
import cv2
import numpy as np
import matplotlib.pyplot as plt

def calc_gray_hist(image):
    rows, cols = image.shape[:2]
    gray_hist = np.zeros([256], np.uint64)
    for i in range(rows):
        for j in range(cols):
            gray_hist[image[i][j]] += 1
    return gray_hist

def otsu_thresh(image):
    rows, cols = image.shape[:2]
    # 计算灰度直方图
    gray_hist = calc_gray_hist(image)
    # 归一化灰度直方图
    norm_hist = gray_hist / float(rows*cols)
    # 计算零阶累积矩, 一阶累积矩
    zero_cumu_moment = np.zeros([256], np.float32)
    one_cumu_moment = np.zeros([256], np.float32)
    for i in range(256):
        if i == 0:
            zero_cumu_moment[i] = norm_hist[i]
            one_cumu_moment[i] = 0
        else:
            zero_cumu_moment[i] = zero_cumu_moment[i-1] + norm_hist[i]
            one_cumu_moment[i] = one_cumu_moment[i - 1] + i * norm_hist[i]
    # 计算方差，找到最大的方差对应的阈值
    mean = one_cumu_moment[255]
    thresh = 0
    sigma = 0
    for i in range(256):
        if zero_cumu_moment[i] == 0 or zero_cumu_moment[i] == 1:
            sigma_tmp = 0
        else:
            sigma_tmp = math.pow(mean*zero_cumu_moment[i] - one_cumu_moment[i], 2) / (zero_cumu_moment[i] * (1.0-zero_cumu_moment[i]))
        if sigma < sigma_tmp:
            thresh = i
            sigma = sigma_tmp
    # 阈值分割
    thresh_img = image.copy()
    thresh_img[thresh_img>thresh] = 255
    thresh_img[thresh_img<=thresh] = 0
    return thresh, thresh_img


# if __name__ == '__main__':
#     image = cv2.imread('E:\picture\lenna.png', 0)
#     gray_hist = calc_gray_hist(image)
#     thresh, thresh_img = otsu_thresh(image)
#     print(thresh)
#     cv2.imwrite('E:\picture\otsu.jpg', thresh_img)
#     cv2.imshow('thresh', thresh_img)
#     cv2.waitKey()
#     x = range(len(gray_hist))
#     plt.plot(x,gray_hist)
#     plt.show()
#     # cv2.imshow('thresh1', gray_hist_)
#     cv2.waitKey(0)
# ////////////////////////////////////////////////////////////////////////
# //////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////    

# Gray scale
def BGR2GRAY(img):
    b = img[:, :, 0].copy()
    g = img[:, :, 1].copy()
    r = img[:, :, 2].copy()

    # Gray scale
    out = 0.2126 * r + 0.7152 * g + 0.0722 * b
    out = out.astype(np.uint8)

    return out

# Otsu Binalization
def otsu_binarization(img, th=128):
    H, W = img.shape
    out = img.copy()

    max_sigma = 0
    max_t = 0

    # determine threshold
    for _t in range(1, 255):
        v0 = out[np.where(out < _t)]
        m0 = np.mean(v0) if len(v0) > 0 else 0.
        w0 = len(v0) / (H * W)
        v1 = out[np.where(out >= _t)]
        m1 = np.mean(v1) if len(v1) > 0 else 0.
        w1 = len(v1) / (H * W)
        sigma = w0 * w1 * ((m0 - m1) ** 2)
        if sigma > max_sigma:
            max_sigma = sigma
            max_t = _t

    # Binarization
    print("threshold >>", max_t)
    th = max_t
    out[out < th] = 0
    out[out >= th] = 255

    return out


# Morphology Dilate
def Morphology_Dilate(img, Dil_time=1):
    H, W = img.shape

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each dilate time
    out = img.copy()
    for i in range(Dil_time):
        tmp = np.pad(out, (1, 1), 'edge')
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) >= 255:
                    out[y, x] = 255

    return out


# Morphology Erode
def Morphology_Erode(img, Erode_time=1):
    H, W = img.shape
    out = img.copy()

    # kernel
    MF = np.array(((0, 1, 0),
                (1, 0, 1),
                (0, 1, 0)), dtype=np.int)

    # each erode
    for i in range(Erode_time):
        tmp = np.pad(out, (1, 1), 'edge')
        # erode
        for y in range(1, H):
            for x in range(1, W):
                if np.sum(MF * tmp[y-1:y+2, x-1:x+2]) < 255*4:
                    out[y, x] = 0

    return out

img = cv2.imread("E:\picture\lenna.png")
gray = BGR2GRAY(img)
otsu = otsu_binarization(gray)
erode_result = Morphology_Erode(otsu, Erode_time=2)
dilate_result = Morphology_Dilate(otsu,Dil_time=2)
cv2.imwrite("E:\picture\Black_and_white.jpg",otsu)
cv2.imshow("Black_and_white",otsu)
cv2.imwrite("E:\picture\erode_result.jpg", erode_result)
cv2.imshow("erode_result", erode_result)
cv2.imwrite("E:\picture\dilate_result.jpg", dilate_result)
cv2.imshow("dilate_result",dilate_result)
cv2.waitKey(0)
cv2.destroyAllWindows()







