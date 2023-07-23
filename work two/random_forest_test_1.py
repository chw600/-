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
def compli(img):
    if img.shape != (256, 256):
        r, c = img.shape
        img = np.pad(img, [(int((256 - r) / 2), int((256 - r + 1) / 2)),
                           (int((256 - c) / 2), int((256 - c + 1) / 2))], 'constant')
    return img

def fearure(img):
    img = np.array(img)
    retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    img = cv2.filter2D(img, -1, retval)
    img = img.flatten()
    #hu_result = cv2.HuMoments(img)
    #return hu_result
    return img

if __name__ == '__main__':
    x_train = np.zeros(shape=(500, 196608))
    y_train = np.zeros(shape=(500))
    z_train = np.zeros(shape=(500))
    x_valid = np.zeros(shape=(200, 196608))
    y_valid = np.zeros(shape=(200))
    z_valid = np.zeros(shape=(500))
    #y_train = np.float32(y_train)
    #y_valid = np.float32(y_valid)
    y_train = y_train.astype(np.str)
    y_valid = y_valid.astype(np.str)
    x_test = np.zeros(shape=(30, 196608))
    k = ['bo_', 'chu_', 'gong_', 'hang_', 'huai_']
    for i in range(5):
        for j in range(100):
            p = 100 * i + j + 1
            c = 'training/' + k[i] + str(p) + '.png'
            #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(c)
            img = cv2.resize(src=img, dsize=(256, 256))
            img = np.array(img)
            hu_img = img.reshape(1, 196608)
            #hu_img = hu_img.reshape(7)
            x_train[p - 1] = hu_img
            y_train[p - 1] = k[i]
            #z_train[p - 1] = k[i]

    for i in range(30):
        p = i + 1
        c = 'testing/' + str(p) + '.png'
        #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(c)
        img = cv2.resize(src=img, dsize=(256, 256))
        img = np.array(img)
        hu_img = img.reshape(1, 196608)
        #hu_img = hu_img.reshape(7)
        x_test[p - 1] = hu_img

    for i in range(1, 5):
        for j in range(40):
            p = 40 * (i - 1) + j + 1
            c = 'validation/' + k[i] + str(p) + '.png'
            #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(c)
            img = cv2.resize(src=img, dsize=(256, 256))
            img = np.array(img)
            hu_img = img.reshape(1, 196608)
            #hu_img = hu_img.reshape(7)
            x_valid[p - 1] = hu_img
            y_valid[p - 1] = k[i]
            #z_valid[p - 1] = k[i]
    for j in range(40):
        c = 'validation/' + k[0] + str(160 + j + 1) + '.png'
        #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(c)
        img = cv2.resize(img, (256, 256))
        img = np.array(img)
        hu_img = img.reshape(1, 196608)
        #hu_img = hu_img.reshape(7)
        x_valid[160 + j] = hu_img
        y_valid[160 + j] = k[0]
        #z_valid[160 + j] = k[0]

    clf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=1)
    clf.fit(x_train, y_train)
    preds1 = clf.predict(x_train)
    preds2 = clf.predict(x_valid)

    print('train的正确率是：', clf.score(x_train, y_train))
    print('valid的正确率是：', clf.score(x_valid, y_valid))

    #km = OneVsRestClassifier(svm.SVC(C=1.2, gamma='auto', verbose=True))
    #km.fit(x_train, y_train)
    #preds3 = km.predict(x_train)
    #preds4 = km.predict(x_valid)

    #print('train的正确率是：', km.score(x_train, y_train))
    #print('valid的正确率是：', km.score(x_valid, y_valid))