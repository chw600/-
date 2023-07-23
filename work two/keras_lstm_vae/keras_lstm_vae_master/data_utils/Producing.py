import numpy as np
import torch
import os

Testpath = "../ServerMachineDatatset/test"
Test_labelpath = "../Producing_data/test_label/machine-1-3.txt"

# name = "machine-" + str(1) + "-" + str(3)
# print(os.path.exists("../ServerMachineDataset/test/machine-1-3.txt"))
Testset = np.loadtxt("../ServerMachineDataset/test/machine-1-3.txt", delimiter=',')
Test_label = np.zeros(Testset.shape, dtype=float)
for i in range(393, 417):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(1258, 1269):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(4873, 4876):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(5614, 5662):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(11311, 11518):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(13872, 13916):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(17017, 17242):
    Test_label[i][12] = 1
    Test_label[i][15] = 1
for i in range(13351, 13381):
    Test_label[i][1] = 1
    Test_label[i][19] = 1
    Test_label[i][20] = 1
    Test_label[i][21] = 1
    Test_label[i][22] = 1
    Test_label[i][23] = 1
    Test_label[i][24] = 1
    Test_label[i][25] = 1
    Test_label[i][26] = 1
    Test_label[i][28] = 1
    Test_label[i][31] = 1
    Test_label[i][32] = 1
    Test_label[i][35] = 1
    Test_label[i][36] = 1
for i in range(16221, 16282):
    Test_label[i][1] = 1
    Test_label[i][19] = 1
    Test_label[i][20] = 1
    Test_label[i][21] = 1
    Test_label[i][22] = 1
    Test_label[i][23] = 1
    Test_label[i][24] = 1
    Test_label[i][25] = 1
    Test_label[i][26] = 1
    Test_label[i][28] = 1
    Test_label[i][31] = 1
    Test_label[i][32] = 1
    Test_label[i][35] = 1
    Test_label[i][36] = 1
for i in range(17661, 17717):
    Test_label[i][19] = 1
    Test_label[i][20] = 1
    Test_label[i][21] = 1
    Test_label[i][22] = 1
    Test_label[i][23] = 1
    Test_label[i][30] = 1
    Test_label[i][33] = 1
    Test_label[i][34] = 1
for i in range(22924, 23020):
    Test_label[i][1] = 1
    Test_label[i][2] = 1
    Test_label[i][3] = 1
    Test_label[i][4] = 1
    Test_label[i][5] = 1
    Test_label[i][6] = 1
    Test_label[i][7] = 1
    Test_label[i][8] = 1
    Test_label[i][9] = 1
    Test_label[i][10] = 1
    Test_label[i][11] = 1
    Test_label[i][12] = 1
    Test_label[i][13] = 1
    Test_label[i][14] = 1
    Test_label[i][15] = 1
    Test_label[i][16] = 1
    Test_label[i][17] = 1
    Test_label[i][18] = 1
    Test_label[i][19] = 1
    Test_label[i][20] = 1
    Test_label[i][21] = 1
    Test_label[i][22] = 1
    Test_label[i][23] = 1
    Test_label[i][24] = 1
    Test_label[i][25] = 1
    Test_label[i][26] = 1
    Test_label[i][28] = 1
    Test_label[i][29] = 1
    Test_label[i][30] = 1
    Test_label[i][31] = 1
    Test_label[i][32] = 1
    Test_label[i][33] = 1
    Test_label[i][34] = 1
    Test_label[i][35] = 1
    Test_label[i][36] = 1
print(Test_label.shape)
np.savetxt(Test_labelpath, Test_label, delimiter=',')
