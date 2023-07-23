
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from PIL import Image,ImageOps
import math


def pixel_probability(img):
    """
    计算像素值出现概率
    :param img:
    :return:
    """
    img = np.array(img)
    prob = np.zeros(shape=(256))
    r, c = img.shape
    for rv in range(r):
        for cv in range(c):
            prob[img[rv][cv]] += 1
    prob = prob / (r * c)
    return prob

def img_to_log(img):
    img = np.array(img)
    r, c = img.shape
    for i in range(r):
        for j in range(c):
            if img[i][j] == 0:
                img[i][j]=1
            img[i][j] = np.log2(img[i][j] + 1e-10)
    return img

def log_to_img(img):
    img = np.array(img)
    r, c = img.shape
    for i in range(2000):
        for j in range(1500):
            img[i][j] = 2 ** img[i][j]
    return img

def probability_to_histogram(img, prob):
    """
    根据像素概率将原始图像直方图均衡化
    :param img:
    :param prob:
    :return: 直方图均衡化后的图像
    """
    prob = np.cumsum(prob)
    img_map = [int(8 * prob[i]) for i in range(8)]
    img = np.array(img)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]
    return img

def calPSNR(img1, img2):
    mse = 0
    img1=img1.flatten()
    img2=img2.flatten()
    temp=img1-img2
    for i in range(3000000):
        mse+=abs(temp[i]) ** 2
    mse=mse/3000000
    psnr = 10 * math.log10(pow(255, 2) / mse)
    return psnr

if __name__ == '__main__':

    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    img = Image.open('./picture/0.jpg')
    img = img.convert('L')
    im = Image.open('./picture/0.jpg')
    im = img.convert('L')
    prob1 = pixel_probability(img)
    img = img_to_log(img)
    prob3 = pixel_probability(img)
    print(img)
    img = probability_to_histogram(img, prob3)
    img = log_to_img(img)
    prob2 = pixel_probability(img)
    plt.figure(figsize=(7, 5))
    plt.subplot(121)
    im1=plt.bar([i for i in range(256)], prob1, width=1), plt.title('处理前图片', fontproperties=font)
    plt.subplot(122)
    im2 = plt.bar([i for i in range(256)], prob2, width=1), plt.title('处理后图片', fontproperties=font)
    plt.savefig("source_hist2.jpg")
    plt.figure('对比图', figsize=(7, 5))
    plt.subplot(121)
    plt.imshow(im, plt.cm.gray), plt.title('处理前图片', fontproperties=font)
    plt.subplot(122)
    plt.imshow(img, plt.cm.gray), plt.title('处理后图片', fontproperties=font)
    plt.savefig("source_hist.jpg")

    plt.show()
    im=np.array(im,dtype=object)
    img3=np.array(img,dtype=object)
    psnr1 = calPSNR(im, img3)
    print(psnr1)