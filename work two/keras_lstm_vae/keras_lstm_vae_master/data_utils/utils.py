import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,accuracy_score, f1_score, recall_score, precision_score
import matplotlib.pyplot as plt
import torch

def compute_score(x, preds):

    scores = torch.pow(x.view(-1), 2) - torch.pow(preds.view(-1), 2)

    return scores

def lstm_evaluate(actual_values, predicted_values, actual_labels):

    best_F1_Score = 0
    best_threshold = 0
    test_accuracy = 0
    test_precision = 0
    test_recall = 0
    scaler = MinMaxScaler(feature_range=(0,1))
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
