from matplotlib import pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
import cv2

im = cv2.imread('./picture/0.jpg',0)
img = cv2.imread('./picture/0.jpg',0)
h, w = img.shape
prob = np.zeros(shape=(256))
for i in range(h):
    for j in range(w):
        prob[img[i][j]]+=1
prob = prob/(h*w)
prob = np.cumsum(prob)
img_map = [int(255*prob[i]) for i in range(256)]
for i in range(h):
    for j in range(w):
        img[i, j] = img_map[img[i, j]]
font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
plt.figure()
plt.bar([i for i in range(256)], img, width=1)
#plt.subplot(121)  # 子图1
#plt.imshow(im1), plt.title('原图直方图', fontproperties=font)
#im2 = plt.bar([i for i in range(256)], img, width=1)
#plt.subplot(122)  # 子图1
#plt.imshow(im2), plt.title('直方图均衡化结果', fontproperties=font)

plt.figure('对比图', figsize=(7, 5))
plt.subplot(121)
plt.imshow(im, plt.cm.gray), plt.title('处理前图片', fontproperties=font)
plt.subplot(122)
plt.imshow(img, plt.cm.gray), plt.title('处理后图片', fontproperties=font)

plt.show()