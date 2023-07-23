import numpy as np
import os

P_2_train = "keras_lstm_vae/keras_lstm_vae_master/SMAP/train_selective/P-2.npy"
P_2_test = "keras_lstm_vae/keras_lstm_vae_master/SMAP/test_selective/P-2.npy"
P_2_label_path = "keras_lstm_vae/keras_lstm_vae_master/SMAP/label_selective/P-2.npy"
Trainpath = "keras_lstm_vae/keras_lstm_vae_master/ServerMachineDataset/train"
Testpath = "keras_lstm_vae/keras_lstm_vae_master/ServerMachineDataset/test"
Test_labelpath = "keras_lstm_vae/keras_lstm_vae_master/ServerMachineDataset/test_label"
Interpretationpath = "keras_lstm_vae/keras_lstm_vae_master/ServerMachineDataset/interpretation_label"
# Trainset = {}
# for i in range(1, 4):
#     for j in range(1, i+8):
#         name = "machine-" + str(i) + "-" + str(j)
#         Trainset[name] = np.loadtxt(Trainpath + "/" + name + ".txt", delimiter=',')
# Trainset["machine-3-11"] = np.loadtxt(Trainpath + "/machine-3-11.txt", delimiter=',')
# # print(Trainset["machine-1-3"])
# print(Trainset["machine-1-3"].shape)
# # print(Trainset["machine-1-4"])
# print(Trainset["machine-1-4"].shape)
# # print(Trainset["machine-1-6"])
# print(Trainset["machine-1-6"].shape)
# # print(Trainset["machine-1-8"])
# print(Trainset["machine-1-8"].shape)

P_2_train = np.load(P_2_train)
print(P_2_train.shape)
P_2_test = np.load(P_2_test)
print(P_2_test.shape)
P_2_label = np.ones(shape=(P_2_test.shape[0], 1))
for i in range(5300, 6576):
    P_2_label[i] = 0
# for i in range(5200, 5301):
#     T_3_label[i] = 0
# for i in range(6449, 6570):
#     T_1_label[i] = 0
np.save(P_2_label_path, P_2_label)

# Interpretationset = {}
# for i in range(1, 4):
#     for j in range(1, i+8):
#         name = "machine-" + str(i) + "-" + str(j)
#         Interpretationset[name] = np.loadtxt(Interpretationpath + "/" + name + ".txt", delimiter={':', ',', '-'})
# Interpretationset["machine-3-11"] = np.loadtxt(Interpretationpath + "/machine-3-11.txt", delimiter={':', ',', '-'})
# # print(Trainset["machine-1-3"])
# print(Interpretationset["machine-1-3"].shape)
# # print(Trainset["machine-1-4"])
# print(Interpretationset["machine-1-4"].shape)
# # print(Trainset["machine-1-6"])
# print(Interpretationset["machine-1-6"].shape)
# # print(Trainset["machine-1-8"])
# print(Interpretationset["machine-1-8"].shape)