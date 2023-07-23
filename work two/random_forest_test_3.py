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
import csv
from tensorflow import keras
import math

def write_csv(id, pred):
    path = "result.csv"
    with open(path, 'w', newline='', encoding='utf8') as f:
        csv_write = csv.writer(f)

        for pair in zip(id, pred):
            csv_write.writerow(pair)


if __name__ == '__main__':

    k = np.array(['bo', 'chu', 'gong', 'hang', 'huai'])
    x_train = np.loadtxt('data/x_train_.txt', dtype='int')
    y_train = np.loadtxt('data/y_train_.txt', dtype='str')
    x_valid, y_valid = {}, {}
    for i in range(5):
        x_valid_path = 'data/x_valid_' + str(i) + '.txt'
        y_valid_path = 'data/y_valid_' + str(i) + '.txt'
        x_valid['x_valid_' + str(i)] = np.loadtxt(x_valid_path, dtype='int')
        y_valid['y_valid_' + str(i)] = np.loadtxt(y_valid_path, dtype='str')
    x_valid_new = np.concatenate((x_valid['x_valid_0'], x_valid['x_valid_1'],
                                  x_valid['x_valid_2'], x_valid['x_valid_3'], x_valid['x_valid_4']))
    y_valid_new = np.concatenate((y_valid['y_valid_0'], y_valid['y_valid_1'],
                                  y_valid['y_valid_2'], y_valid['y_valid_3'], y_valid['y_valid_4']))
    clf = RandomForestClassifier(n_estimators=300, max_depth=15, min_samples_leaf=1)
    clf.fit(x_train, y_train)
    print('train的正确率是：', clf.score(x_train, y_train))
    f = []
    for i in range(5):
        print('valid_'+ k[i]+'的正确率是：', clf.score(x_valid['x_valid_' + str(i)], y_valid['y_valid_' + str(i)]))
        f.append(clf.score(x_valid['x_valid_' + str(i)], y_valid['y_valid_' + str(i)]))
        plt.figure()
        plt.plot(y_valid['y_valid_' + str(i)], label='y_valid_'+ k[i])
        plt.plot(clf.predict(x_valid['x_valid_' + str(i)]), color='red', label='predict_'+ k[i] )
        plt.legend(loc='upper left')
        plt.title('y_valid-predict_'+ k[i])
        plt.show()
        plt.savefig('result/y_valid-predict_' + k[i] + '.jpg')
    f = np.array(f)
    plt.figure()
    plt.bar(k, f)
    plt.ylim(0, 1)
    plt.title('score')
    plt.show()
    plt.savefig('result/score_bar.jpg')
    print('valid的正确率是：', clf.score(x_valid_new, y_valid_new))
    id = np.array([i for i in range(30)],dtype='str')
    x_test_path = 'data/x_test.txt'
    x_test = np.loadtxt(x_test_path, dtype='int')
    pred = clf.predict(x_test)
    write_csv(id, pred)