import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import prewitt_h,prewitt_v
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
import cv2
from sklearn import neighbors
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import neighbors
from tensorflow import keras
from PIL import Image
import math

def fearure(img):
    retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
    img = np.array(img)
    #img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    #img = cv2.filter2D(img, -1, retval)
    img = img.flatten()
    #hu_result = cv2.HuMoments(img)
    #return hu_result
    return img


if __name__ == '__main__':
    x_train = np.zeros(shape=(500, 65536))
    y_train = np.zeros(shape=(500))
    y_train = y_train.astype(np.str)
    x_test = np.zeros(shape=(30, 65536))
    k = ['bo_', 'chu_', 'gong_', 'hang_', 'huai_']
    for i in range(5):
        for j in range(100):
            p = 100*i+j+1
            c = 'training/'+k[i]+str(p)+'.png'
            retval = cv2.getGaborKernel(ksize=(4,4),sigma=10,theta=60,lambd=10,gamma=1.2)
            img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if img.shape != (256, 256):
                r, c = img.shape
                img = np.pad(img, [(int((256-r)/2), int((256-r+1)/2)), (int((256 - c)/2), int((256 - c +1)/2))], 'constant')
            hu_img = fearure(img)
            #hu_img = hu_img.reshape(7)
            x_train[p-1] = hu_img
            y_train[p-1] = k[i]

    for i in range(30):
        p = i + 1
        c = 'testing/' + str(p)+'.png'
        img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        if img.shape != (256, 256):
            r, c = img.shape
            img = np.pad(img,[(int((256 - r) / 2), int((256 - r + 1) / 2)), (int((256 - c) / 2), int((256 - c + 1) / 2))],
                         'constant')
        hu_img = fearure(img)
        #hu_img = hu_img.reshape(7)
        x_test[p - 1] = hu_img
    #km = OneVsRestClassifier(svm.SVC(C=1.2, gamma='auto', verbose=True))
    km = neighbors.KNeighborsClassifier(n_neighbors=5)
    km.fit(x_train, y_train)
    t=0
    for i in range(1, 5):
        for j in range(40):
            p = 40 * (i - 1) + j + 1
            c = 'validation/' + k[i] + str(p) + '.png'
            retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
            img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if img.shape != (256, 256):
                r, c = img.shape
                img = np.pad(img, [(int((256 - r) / 2), int((256 - r + 1) / 2)),
                                   (int((256 - c) / 2), int((256 - c + 1) / 2))], 'constant')
            hu_img = fearure(img)
            #hu_img = hu_img.reshape(1, 7)
            x_predict = km.predict([hu_img])
            if x_predict == k[i]:
                t += 1
    print('accuracy=', t/160)
    for j in range(40):
        c = 'validation/' + k[0] + str(160 + j + 1) + '.png'
        retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
        img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        if img.shape != (256, 256):
            r, c = img.shape
            img = np.pad(img,
                         [(int((256 - r) / 2), int((256 - r + 1) / 2)), (int((256 - c) / 2), int((256 - c + 1) / 2))],
                         'constant')
        x_predict = ['a', 'b', 'c']
        hu_img = fearure(img)
        #hu_img = hu_img.reshape(1, 7)
        x_predict = km.predict([hu_img])
        if x_predict == k[i]:
            t += 1
    print('accuracy=', t/200)
