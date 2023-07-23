import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager
from sklearn import svm
from sklearn import metrics
import datetime
from torch.utils.tensorboard import SummaryWriter

# Trainpath = "ServerMachineDataset/train"
# Testpath = "ServerMachineDataset/test"
# Test_labelpath = "ServerMachineDataset/test_label"
# Interpretationpath = "ServerMachineDataset/interpretation_label"
#
# X_train = np.loadtxt(Trainpath + "/machine-3-11.txt", delimiter=',')
# X_test = np.loadtxt(Testpath + "/machine-3-11.txt", delimiter=',')
# Y_label = np.loadtxt(Test_labelpath + "/machine-3-11.txt", delimiter=',')
file_path = 'SMAP/train_selective/T-3.npy'
test_file_path = "SMAP/test_selective/T-3.npy"
test_label_path = "SMAP/label_selective/T-3.npy"
X_train = np.load(file_path)
X_test = np.load(test_file_path)
Y_label = np.load(test_label_path)
Y_label = np.reshape(Y_label, (Y_label.shape[0]))
# fit the model
Y_train_label = np.ones(shape=(X_train.shape[0]))
clf = svm.OneClassSVM(nu=0.1, kernel='rbf', gamma=0.1)
clf.fit(X_train, Y_train_label)
begin_time = datetime.datetime.now()
Y_test_label = clf.predict(X_test)
Y_test_label = np.where(Y_test_label==1, 1, 0)
TP = 0
FN = 0
FP = 0
# 计算TP和FN
index = []
truth = np.where(Y_label == 1)[0]
tuple = (truth[0],)
for i in range(len(truth) - 1):
    if truth[i] + 1 != truth[i + 1]:
        tuple = tuple + (truth[i],)
        index.append(tuple)
        tuple = (truth[i + 1],)
for i in index:
    if sum(Y_label[i[0]:i[1]] * Y_test_label[i[0]:i[1]]) > 0:  # 实际为异常值，且检测到了异常值
        TP = TP + 1
    else:  # 实际为异常值，但是没有检测到异常值
        FN = FN + 1
        # 计算FP
        index = []
        labels = 1 - Y_label
        truth = np.where(Y_label == 1)[0]
        tuple = (truth[0],)
for i in range(len(truth) - 1):
    if truth[i] + 1 != truth[i + 1]:
        tuple = tuple + (truth[i],)
        index.append(tuple)
        tuple = (truth[i + 1],)
for i in index:
    if sum(Y_label[i[0]:i[1]] * Y_test_label[i[0]:i[1]]) > 0:  # 实际为正常值，检测成了异常值
        FP = FP + 1
precision = TP / (TP + FP + 0.00001)
recall = TP / (TP + FN + 0.00001)
f1 = 2 * precision * recall / (precision + recall + 0.00001)
end_time = datetime.datetime.now()
log_dir = 'log/'
writer = SummaryWriter(log_dir)
print("f1:", f1, "acc:", precision, "rec:", recall, "time:", end_time -  begin_time)
with open('experiment_data/Ocsvm_Evaluate_SMAP.txt', 'a') as f:
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    f.write("timestamp: {0}\n".format(timestamp))
    f.write("数据集: {}\n".format("SMAP_T_3"))
    f.write('模型:{}\n'.format("Ocsvm"))
    f.write("Precision: {0:.2%}".format(precision))
    f.write("Recall: {0:.2%}".format(recall))
    f.write("F1-Score: {0:.2%}\n".format(f1))
    f.write("second: {0}\n".format((end_time - begin_time).seconds))
    f.write("micro: {0}\n".format((end_time - begin_time).microseconds))
writer.close()
# print("f1:", f1, "acc:", acc)