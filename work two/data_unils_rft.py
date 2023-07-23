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
import math

def feature(img):
    img = np.array(img)
    ht = np.zeros(shape=256)
    r,t = img.shape
    retval = cv2.getGaborKernel(ksize=(4, 4), sigma=10, theta=60, lambd=10, gamma=1.2)
    img = cv2.GaussianBlur(img, (3, 3), 0, 0)
    #img = cv2.filter2D(img, -1, retval)
    for i in range(r):
        for j in range(t):
            ht[img[i][j]] += 1
    #hu_result = cv2.HuMoments(img)
    #return hu_result
    return ht

if __name__ == '__main__':
    x_train = np.zeros(shape=(500, 768))
    y_train = np.array(['xxxx'] * 500, dtype='str')
    x_valid = np.zeros(shape=(200, 768))
    y_valid = np.array(['xxxx'] * 200, dtype='str')
    x_test = np.zeros(shape=(30, 768))
    k = ['bo', 'chu', 'gong', 'hang', 'huai']
    for i in range(5):
        for j in range(100):
            p = 100 * i + j + 1
            c = 'training/' + k[i] + '_' + str(p) + '.png'
            #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(c)
            img = cv2.resize(src=img, dsize=(256, 256))
            img = np.array(img)
            img = img.transpose(2, 0, 1)
            #R = cv2.split(img)[0]
            #G = cv2.split(img)[1]
            #B = cv2.split(img)[2]
            #hu_img = np.concatenate((feature(R), feature(G), feature(B)), axis=0)
            hu_img = np.concatenate((feature(img[0]), feature(img[1]), feature(img[2])), axis=0)
            #hu_img = hu_img.reshape(7)
            x_train[(p - 1)] = hu_img
            y_train[(p - 1)] = k[i]
    #x_train_path = 'data/x_train_2.txt'
    #y_train_path = 'data/y_train_2.txt'
    x_train_path = 'data/x_train_.txt'
    y_train_path = 'data/y_train_.txt'
    np.savetxt(x_train_path, x_train, fmt='%d')
    np.savetxt(y_train_path, y_train, fmt='%s')


    for i in range(30):
        p = i + 1
        c = 'testing/' + str(p) + '.png'
        #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(c)
        img = cv2.resize(src=img, dsize=(256, 256))
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        #R = cv2.split(img)[0]
        #G = cv2.split(img)[1]
        #B = cv2.split(img)[2]
        #hu_img = np.concatenate((feature(R), feature(G), feature(B)), axis=0)
        hu_img = np.concatenate((feature(img[0]), feature(img[1]), feature(img[2])), axis=0)
        x_test[(p - 1)] = hu_img
    #x_test_path = 'data/x_test_2.txt'
    x_test_path = 'data/x_test.txt'
    np.savetxt(x_test_path, x_test, fmt='%d')

    for i in range(1, 5):
        for j in range(40):
            p = 40 * (i - 1) + j + 1
            c = 'validation/' + k[i] + '_' + str(p) + '.png'
            #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
            img = cv2.imread(c)
            img = cv2.resize(src=img, dsize=(256, 256))
            img = np.array(img)
            img = img.transpose(2, 0, 1)
            #R = cv2.split(img)[0]
            #G = cv2.split(img)[1]
            #B = cv2.split(img)[2]
            #hu_img = np.concatenate((feature(R), feature(G), feature(B)), axis=0)
            hu_img = np.concatenate((feature(img[0]), feature(img[1]), feature(img[2])), axis=0)

            x_valid[(p - 1)] = hu_img
            y_valid[(p - 1)] = k[i]
        #x_valid_path = 'data/x_valid_2' + str(i) + '.txt'
        #y_valid_path = 'data/y_valid_2' + str(i) + '.txt'
        x_valid_path = 'data/x_valid_' + str(i) + '.txt'
        y_valid_path = 'data/y_valid_' + str(i) + '.txt'
        np.savetxt(x_valid_path, x_valid[i:i + 40], fmt='%d')
        np.savetxt(y_valid_path, y_valid[i:i + 40], fmt='%s')
    for j in range(40):
        c = 'validation/' + k[0] + '_' + str(160 + j + 1) + '.png'
        #img = cv2.imread(c, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(c)
        img = cv2.resize(img, (256, 256))
        img = np.array(img)
        img = img.transpose(2, 0, 1)
        #R = cv2.split(img)[0]
        #G = cv2.split(img)[1]
        #B = cv2.split(img)[2]
        #hu_img = np.concatenate((feature(R), feature(G), feature(B)), axis=0)
        hu_img = np.concatenate((feature(img[0]), feature(img[1]), feature(img[2])), axis=0)
        x_valid[160 + j] = hu_img
        y_valid[160 + j] = k[0]
    #x_valid_path = 'data/x_valid_2' + str(0) + '.txt'
    #y_valid_path = 'data/y_valid_2' + str(0) + '.txt'
    x_valid_path = 'data/x_valid_' + str(0) + '.txt'
    y_valid_path = 'data/y_valid_' + str(0) + '.txt'
    np.savetxt(x_valid_path, x_valid[0: 40], fmt='%d')
    np.savetxt(y_valid_path, y_valid[0: 40], fmt='%s')