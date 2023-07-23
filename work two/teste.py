import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import prewitt_h,prewitt_v
import sklearn
import pandas as pd
import os
import sys
import tensorflow as tf
import cv2
from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier

from tensorflow import keras
from PIL import Image
import math
x = [[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
y = [1,2,3,4,5,6,7,8,9,10,11,12]
z = [[[1,2],[3,4]],[[5,6],[7,8]],[[9,10],[11,12]]]
x = np.array(x)
y = np.array(y)
z = np.array(z)
print(z)
t = np.concatenate((x,x),axis=0)
print(t)
print(y.shape)