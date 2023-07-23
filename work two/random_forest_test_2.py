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
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn import neighbors

from tensorflow import keras
from PIL import Image
import math

def feature(img):
    img = np.array(img)
    #retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    #img = cv2.filter2D(img, -1, retval)
    img = cv2.calcHist(img,[1,2],None,[256,256],[0.0,255,0.0,255])
    #hu_result = cv2.HuMoments(img)
    #return hu_result
    return img

if __name__ == '__main__':
    x_train = np.zeros(shape=(500, 65536))
    y_train = np.array(['xxxx'] * 500, dtype='str')
    x_valid = np.zeros(shape=(200, 65536))
    y_valid = np.array(['xxxx'] * 200, dtype='str')
    x_test = np.zeros(shape=(30, 65536))
    k = ['bo_', 'chu_', 'gong_', 'hang_', 'huai_']
    for i in range(5):
        for j in range(100):
            p = 100 * i + j + 1
            c = 'training/' + k[i] + str(p) + '.png'
            # img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(c)
            img = cv2.resize(src=img, dsize=(256, 256))
            img = np.array(img)
            hu_img = feature(img)
            hu_img = hu_img.flatten()
            # hu_img = hu_img.reshape(7)
            x_train[(p - 1)] = hu_img
            y_train[(p - 1)] = k[i]

    for i in range(30):
        p = i + 1
        c = 'testing/' + str(p) + '.png'
        # img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(c)
        img = cv2.resize(src=img, dsize=(256, 256))
        img = np.array(img)
        hu_img = feature(img)
        hu_img = hu_img.flatten()

        x_test[(p - 1)] = hu_img

    for i in range(1, 5):
        for j in range(40):
            p = 40 * (i - 1) + j + 1
            c = 'validation/' + k[i] + str(p) + '.png'
            # img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(c)
            img = cv2.resize(src=img, dsize=(256, 256))
            img = np.array(img)
            hu_img = feature(img)
            hu_img = hu_img.flatten()

            x_valid[(p - 1)] = hu_img
            y_valid[(p - 1)] = k[i]
    for j in range(40):
        c = 'validation/' + k[0] + str(160 + j + 1) + '.png'
        # img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(c)
        img = cv2.resize(img, (256, 256))
        img = np.array(img)
        hu_img = feature(img)
        hu_img = hu_img.flatten()
        x_valid[160 + j] = hu_img
        y_valid[160 + j] = k[0]

    clf = RandomForestClassifier(n_estimators=120, max_depth=15, min_samples_leaf=1)
    clf.fit(x_train, y_train)
    preds1 = clf.predict(x_train)
    preds2 = clf.predict(x_valid)

    print('train的正确率是：', clf.score(x_train, y_train))
    print('valid的正确率是：', clf.score(x_valid, y_valid))