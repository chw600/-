import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import pyplot as plt

from matplotlib.font_manager import FontProperties
from skimage import exposure

# 读入图片
im = cv2.imread('./picture/0.jpg' , 0)
img = cv2.imread('./picture/0.jpg' , 0)
# 如果图片为空，返回错误信息，并终止程序
if im is None:
    print("图片打开失败！")
    exit()
# 中值滤波去噪
medStep = 3  # 设置为3*3的滤波器


h,w = img.shape
hist = cv2.calcHist([img],[0],None,[256],[0,255])#这里返回的是次数
hist[0:255] = hist[0:255]/(h*w)#将直方图归一化，化为概率的形式
def cdf(img):
    img = cv2.imread(img,0)
    #flatten() 将数组变成一维
    hist,bins = np.histogram(img.flatten(),256,[0,256])#计算直方图
    #bins:每个区间的起始点和终点，（257,1）
    #hist：直方图（256.1）
    # 计算累积分布图
    print(hist)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max()/ cdf.max()#
    plt.plot(cdf_normalized, color = 'b')#绘制累积灰度级别曲线
    plt.hist(img.flatten(),256,[0,256], color = 'r')#绘制原始图像的直方图
    plt.xlim([0,256])
    plt.legend(('cdf','histogram'), loc = 'upper left')
    plt.show()
 #第二步得到灰度级概率累积直方图
sum_hist = np.zeros(hist.shape)#用于存放灰度级别概率的累和
for i in range(256):
    sum_hist[i] = sum(hist[0:i+1])#将前i+1个灰度级别的出现概率总和赋值给sum_hist[i]
        # 第三步通过映射函数获得原图像灰度级与均衡后图像的灰度级的映射关系，这里创建映射后的灰度级别排序
equal_hist = np.zeros(sum_hist.shape)
for i in range(256):
    equal_hist[i] = int(((256 - 1) - 0) * sum_hist[i] + 0.5)
#第四步根据第三步的映射关系将灰度图每个像素点的灰度级别替换为映射后的灰度级别，这里是这样换的，equal_hist的索引号相当于原先的灰度级别排序，元素值则是映射后的灰度级别
equal_img = img.copy()#用于存放均衡化后图像的灰度值
for i in range(h):
    for j in range(w):
        equal_img[i,j] = equal_hist[img[i,j]]
    #计算得到均衡化后的直方图
equal_hist = cv2.calcHist([equal_img],[0],None,[256],[0,255])
equal_hist[0:255] = equal_hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式
def sys_equalizehist(img):
    '''
    利用系统自带的函数进行直方图均衡化
    :param img: 待处理图像
    :return:  [equ_img,equ_hist]，返回一个列表，第一个元素是均衡化后的图像，第二个是均衡了的直方图
    '''
    img = cv2.imread('./picture/0.jpg', 0)
    h, w = img.shape
    equ_img = cv2.equalizeHist(img)  # 得到直方图均衡化后的图像
    equ_hist = cv2.calcHist([equ_img], [0], None, [256], [0, 255])  # 得到均衡化后的图像的灰度直方图
    equ_hist[0:255] = equ_hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式
    # res = np.hstack((img,equ)) #stacking images side-by-side#这一行是将两个图像进行了行方向的叠加
    return [img, equ_img, equ_hist]
# 创建绘制原图像和均衡化后图像的对应灰度值变化曲线
def equalWithOrignImg(gray_img, sys_img):
    '''
    :param gray_img: 未均衡图
    :param sys_img: 均衡图
    :return:无返回值
    '''
    # 将图像像素变成一维的
    gray_img = gray_img.ravel()
    sys_img = sys_img.ravel()
    # 返回gray_img像素值从小到大的索引号,对其进行排序
    argOfGrayImg = np.argsort(gray_img)
    gray_img = sorted(gray_img)
    # 根据索引号来取sys_img中的元素
    sys_img_sorted = []
    for i in argOfGrayImg:
        sys_img_sorted.append(sys_img[i])
    plt.plot(gray_img, sys_img_sorted)
    plt.show()
if __name__ == '__main__':
    img = "E:\PYTHON\Image_Processing\colorful_lena.jpg"
    gray_img, sys_img, sys_hist = sys_equalizehist(img)
    equalWithOrignImg(gray_img, sys_img)

def sys_equalizehist(img):
    '''
    利用系统自带的函数进行直方图均衡化
    :param img: 待处理图像
    :return:  [equ_img,equ_hist]，返回一个列表，第一个元素是均衡化后的图像，第二个是均衡了的直方图
    '''
    img = cv2.imread('./picture/0.jpg' , 0)
    h,w = img.shape
    equ_img = cv2.equalizeHist(img)#得到直方图均衡化后的图像
    equ_hist = cv2.calcHist([equ_img],[0],None,[256],[0,255])#得到均衡化后的图像的灰度直方图
    equ_hist[0:255] = equ_hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式
    # res = np.hstack((img,equ)) #stacking images side-by-side#这一行是将两个图像进行了行方向的叠加
    return [equ_img,equ_hist]


def def_equalizehist(img, L=256):
    '''
    根据均衡化原理自定义函数
    :param img: 待处理图像
    :param L: 灰度级别的个数
    :return: [equal_img,equal_hist]返回一个列表，第一个元素是均衡化后的图像，第二个是均衡了的直方图
    '''
    img = cv2.imread('./picture/0.jpg' , 0)

    # 第一步获取图像的直方图
    h, w = img.shape
    hist = cv2.calcHist([img], [0], None, [256], [0, 255])  # 这里返回的是次数
    hist[0:255] = hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式

    # 第二步得到灰度级概率累积直方图
    sum_hist = np.zeros(hist.shape)  # 用于存放灰度级别概率的累和
    for i in range(256):
        sum_hist[i] = sum(hist[0:i + 1])  # 将前i+1个灰度级别的出现概率总和赋值给sum_hist[i]

    # 第三步通过映射函数获得原图像灰度级与均衡后图像的灰度级的映射关系，这里创建映射后的灰度级别排序
    equal_hist = np.zeros(sum_hist.shape)
    for i in range(256):
        equal_hist[i] = int(((L - 1) - 0) * sum_hist[i] + 0.5)

    # 第四步根据第三步的映射关系将灰度图每个像素点的灰度级别替换为映射后的灰度级别，这里是这样换的，equal_hist的索引号相当于原先的灰度级别排序，元素值则是映射后的灰度级别
    equal_img = img.copy()  # 用于存放均衡化后图像的灰度值
    for i in range(h):
        for j in range(w):
            equal_img[i, j] = equal_hist[img[i, j]]
    # 计算得到均衡化后的直方图
    equal_hist = cv2.calcHist([equal_img], [0], None, [256], [0, 255])
    equal_hist[0:255] = equal_hist[0:255] / (h * w)  # 将直方图归一化，化为概率的形式
    return [equal_img, equal_hist]


if __name__ == '__main__':
    img = "colorful_lena.jpg"
    sys_img, sys_hist = sys_equalizehist(img)
    def_img, def_hist = def_equalizehist(img)
    x = np.linspace(0, 255, 256)
    plt.subplot(1, 2, 1), plt.plot(x, sys_hist, '-b')
    plt.subplot(1, 2, 2), plt.plot(x, def_hist, '-r')
    plt.show()
def Contrast_and_Brightness(alpha, beta, img):
    """使用公式f(x)=α.g(x)+β"""
    # α调节对比度，β调节亮度
    blank = np.zeros(img.shape, img.dtype)  # 创建图片类型的零矩阵
    dst = cv2.addWeighted(img, alpha, blank, 1 - alpha, beta)  # 图像混合加权
    return dst
img4 = Contrast_and_Brightness(1.1, 30, img)
cv2.imshow("Contrast", img4)
    # 创建一个窗口
plt.figure('对比图', figsize=(7, 5))
    # 中文字体设置
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)  # 新宋体
    # 显示原图
plt.subplot(121)  # 子图1
    # 显示原图，设置标题和字体
plt.imshow(im, plt.cm.gray), plt.title('处理前图片', fontproperties=font)

    # 显示处理过的图像
plt.subplot(122)  # 子图2
    # 显示处理后的图，设置标题和字体
plt.imshow(img4, plt.cm.gray), plt.title('处理后图片', fontproperties=font)
plt.show()
    # 销毁所有窗口
cv2.destroyAllWindows()
