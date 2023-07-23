from utils import *
import os
import tensorflow as tf
import math
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
from tensorflow import config
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

def plot_figure(index, contami, test_vals, predictions, test_labels, model_name, number):
    figsize = (12, 7)
    plt.figure(figsize=figsize)
    plt.plot(test_vals, label='test_abn')
    plt.plot(predictions, color='red', label='test')
    x = np.where(test_labels == 1)[0]
    y = [test_vals[index] for index in x]
    plt.scatter(x, y, s=25, c='green')
    plt.legend(loc='upper left')
    plt.title("test_abn-test_"+str(number+1).format(model_name, index, contami))
    # 保存图片
    plt.savefig("result/lstm-fig/test_abn.png".format(model_name, index, contami))
    plt.show()

def plot_figure_2(index, contami, test_vals, predictions, test_labels, model_name):
    figsize = (12, 7)
    plt.figure(figsize=figsize)
    plt.plot(test_vals, label='test_abn')
    plt.plot(predictions, color='red', label='train')
    x = np.where(test_labels == 1)[0]
    y = [test_vals[index] for index in x]
    plt.scatter(x, y, s=25, c='green')
    plt.legend(loc='upper left')
    plt.title("test_abn-train".format(model_name, index, contami))
    # 保存图片
    plt.savefig("result/lstm-fig/predicted.png".format(model_name, index, contami))
    plt.show()

def plot_figure_3(index, contami, test_vals, predictions, test_labels, model_name):
    figsize = (12, 7)
    plt.figure(figsize=figsize)
    plt.plot(test_vals, label='test')
    plt.plot(predictions, color='red', label='train')
    x = np.where(test_labels == 1)[0]
    y = [test_vals[index] for index in x]
    plt.scatter(x, y, s=25, c='green')
    plt.legend(loc='upper left')
    plt.title("test-train".format(model_name, index, contami))
    # 保存图片
    plt.savefig("result/lstm-fig/test.png".format(model_name, index, contami))
    plt.show()

def lstm_evaluate(actual_values, predicted_values, actual_labels):

    best_F1_Score = 0
    best_threshold = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    scaler = MinMaxScaler(feature_range=(0, 1))
    r, c, w = actual_values.shape
    dif = np.zeros((r, c), dtype='float')
    for i in range(r):
        for j in range(c):
            dif[i][j] = np.sqrt(mean_squared_error(actual_values[i][j], predicted_values[i][j]))
    dif = dif.transpose((1, 0))
    for i in np.arange(0, 1, 0.01):
        threshold = i
        # threshold = i * (measure_rmse(actual_values[:500], predicted_values[:500]))
        # threshold = 8000
        predicted_labels = np.where(dif > threshold, 1, 0)
        actual_labels = actual_labels.reshape(-1)
        predicted_labels = predicted_labels.reshape(-1)
        test_F1_Score = f1_score(actual_labels, predicted_labels, zero_division=0)
        if test_F1_Score > best_F1_Score:
            best_threshold = threshold
            best_F1_Score = test_F1_Score
            test_accuracy = accuracy_score(actual_labels, predicted_labels)
            test_precision = precision_score(actual_labels, predicted_labels, zero_division=0)
            test_recall = recall_score(actual_labels, predicted_labels, zero_division=0)


    print("Threshold: {0}".format(best_threshold))
    print("Accuracy: {0:.2%}".format(test_accuracy))
    print("Precision: {0:.2%}".format(test_precision))
    print("Recall: {0:.2%}".format(test_recall))
    print("F1-Score: {0:.2%}".format(best_F1_Score))
    # ------ Write results to file ------
    # f.write("数据集: real_{0}_{1}%.csv\n".format(index, contami))
    print('---------- Evaluate is written to file! ----------')

#test_data = np.load('experiment_data/SWaT/test_set.npy')
#test_label = np.load('experiment_data/SWaT/test_label.npy')
#test_data_filename = 'MSL_test/processed/M_test.pkl'
test_data_filename = 'data/shanghai_test_abn.pkl'
F = open(test_data_filename, 'rb')
test_data = pickle.load(F)
time_window, sensor_number, feature_number = test_data.shape
#test_label = np.loadtxt('MSL_test/processed/labels.txt', dtype='int')
test_label_filename = 'data/shanghai_test_label.pkl'
F = open(test_label_filename, 'rb')
test_label = pickle.load(F)
dataset_name = 'shanghai'
lstm_model = {}
lstmpath = {}
k1, k2, k5, k6 = [], [], [], []
# fit and predict dataset
test_data_new = test_data[10:-1, :, :]
for i in range(sensor_number):
    a = test_data[:, i, :]
    b = test_data_new[:, i, :]
    k3, k4 = [], []
    for j in range(time_window-11):
        k3.append(a[j:j + 10, :])
        k4.append(b[j, :])
    k3 = np.array(k3)
    k4 = np.array(k4)
    k5.append(k3)
    k6.append(k4)
k5 = np.array(k5,dtype=float)

k6 = np.array(k6)
k6 = np.expand_dims(k6, axis = -2)

for i in range(sensor_number):
    lstmpath[
        'model_' + str(i)] = 'result' + '/lstm_experiment/' + 'shanghai_' + str(i) +  '_lstm_model.hdf5'
    model = tf.keras.models.load_model(lstmpath['model_' + str(i)])
    k1.append(model.predict(k5[i]))
k1 = np.array(k1)
#for i in range(len(test_data[0])):
    #k1.append(lstm(test_data[i]))
    #k2.append((test_abn_data[i])
#normal_label = np.array([0]*939,dtype='int')
#for i in range(939):
    #if abs(k1[i] - test_abn_data_new[i]) > 0.5:
        #normal_label[i]+=1
test_label_data = test_label[10:-1]
#test_label_data = test_label_data.reshape(-1)
#test_accuracy = accuracy_score(normal_label, test_label_data)
#test_precision = precision_score(normal_label, test_label_data, zero_division=0)
#test_recall = recall_score(normal_label, test_label_data, zero_division=0)
#print("test_accuracy:", test_accuracy)
#print("test_precision:", test_precision)
#print("test_recall:", test_recall)
test_data_new = test_data_new.transpose((1, 0, 2))
lstm_evaluate(test_data_new, k1, test_label_data)
#lstm_evaluate(test_data_new, k1, label_data)
#plot_figure(1, 15, test_abn_data_new[:, i],k1[:, i],test_label_data[:, i],'Lstm',i)