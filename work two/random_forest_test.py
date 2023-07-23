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
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    img = cv2.filter2D(img, -1, retval)
    img = img.flatten()
    #hu_result = cv2.HuMoments(img)
    #return hu_result
    return img

if __name__ == '__main__':
    x_train = np.zeros(shape=(500, 65536))
    y_train = np.zeros(shape=(500))
    z_train = np.zeros(shape=(500))
    x_valid = np.zeros(shape=(200, 65536))
    y_valid = np.zeros(shape=(200))
    z_valid = np.zeros(shape=(500))
    y_train = np.float32(y_train)
    y_valid = np.float32(y_valid)
    z_train = np.asarray(str)
    z_valid = np.asarray(str)
    x_test = np.zeros(shape=(30, 65536))
    k = ['bo_', 'chu_', 'gong_', 'hang_', 'huai_']
    for i in range(5):
        for j in range(100):
            p = 100 * i + j + 1
            c = 'training/' + k[i] + str(p) + '.png'
            retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
            img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if img.shape != (256, 256):
                r, c = img.shape
                img = np.pad(img, [(int((256 - r) / 2), int((256 - r + 1) / 2)),
                                   (int((256 - c) / 2), int((256 - c + 1) / 2))], 'constant')
            hu_img = fearure(img)
            #hu_img = hu_img.reshape(7)
            x_train[p - 1] = hu_img
            y_train[p - 1] = i
            #z_train[p - 1] = k[i]

    for i in range(30):
        p = i + 1
        c = 'testing/' + str(p) + '.png'
        img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        if img.shape != (256, 256):
            r, c = img.shape
            img = np.pad(img,
                         [(int((256 - r) / 2), int((256 - r + 1) / 2)), (int((256 - c) / 2), int((256 - c + 1) / 2))],
                         'constant')
        hu_img = fearure(img)
        #hu_img = hu_img.reshape(7)
        x_test[p - 1] = hu_img

    for i in range(1, 5):
        for j in range(40):
            p = 40 * (i - 1) + j + 1
            c = 'validation/' + k[i] + str(p) + '.png'
            img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            if img.shape != (256, 256):
                r, c = img.shape
                img = np.pad(img, [(int((256 - r) / 2), int((256 - r + 1) / 2)),
                                   (int((256 - c) / 2), int((256 - c + 1) / 2))], 'constant')
            hu_img = fearure(img)
            #hu_img = hu_img.reshape(7)
            x_valid[p - 1] = hu_img
            y_valid[p - 1] = i
            #z_valid[p - 1] = k[i]
    for j in range(40):
        c = 'validation/' + k[0] + str(160 + j + 1) + '.png'
        img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        if img.shape != (256, 256):
            r, c = img.shape
            img = np.pad(img,
                         [(int((256 - r) / 2), int((256 - r + 1) / 2)), (int((256 - c) / 2), int((256 - c + 1) / 2))],
                         'constant')
        hu_img = fearure(img)
        #hu_img = hu_img.reshape(7)
        x_valid[160 + j] = hu_img
        y_valid[160 + j] = 0
        #z_valid[160 + j] = k[0]

    rt = cv2.ml.RTrees_create()
    rt.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 10, 1))
    rt.setCalculateVarImportance(True)
    rt.train(cv2.ml.TrainData_create(np.float32(x_train), layout=0, responses=y_train))
    preds1 = rt.predict(np.float32(x_train), 0)[1]
    preds2 = rt.predict(np.float32(x_valid), 0)[1]
    d=0
    t=0
    threshold = 0.5
    for i in range(len(preds1)):
        if abs(preds1[i] - y_train[i]) < threshold:
            d += 1
    for i in range(len(preds2)):
        if abs(preds2[i] - y_valid[i]) < threshold:
            t += 1
    print('train的正确率是：', d / 500)
    print('valid的正确率是：', t / 200)