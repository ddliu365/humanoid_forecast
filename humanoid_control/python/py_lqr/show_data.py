
# this function for showing data in remote vscode
import numpy as np
import matplotlib.pyplot as plt
import os

rootdir = "/Users/dongdong/proj/data/"
f = []

for folder, subs, files in os.walk(rootdir):
    for filename in files:
        f.append(os.path.abspath(os.path.join(folder, filename)))
    break

for j in f:
    _data = np.load(j)
    num_column = np.size(_data, 1)
    x = np.arange(num_column)
    num_row = np.size(_data, 0)
    for i in range(num_row):
        fig = plt.figure(figsize=(12, 6.5))
        plt.subplots_adjust(wspace=0, hspace=0.5)
        plt.plot(x, _data[i, :])
        plt.title(str(j) + ":" + str(i+1) + "/" + str(num_row))
        plt.ylabel('y')
        plt.xlabel('t')
    plt.show()

'''
# this function for comparing data in vscode remote

root_dir = "/home/jack/catkin_ws/src/humanoid/data/20190620-151635/"
file1_name = "joint_ctrl.npy"
file2_name = "joint_fd.npy"

data_1 = np.load(root_dir + file1_name)
data_2 = np.load(root_dir + file2_name)

num_column = np.size(data_1, 1)
x = np.arange(num_column)
num_row = np.size(data_1, 0)
for i in range(num_row):
    fig = plt.figure(figsize=(12, 6.5))
    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.plot(x, data_1[i, :])
    plt.plot(x, data_2[i, 550:])
    plt.title(file1_name + ":" + str(i+1) + "/" + str(num_row))
    plt.ylabel('y')
    plt.xlabel('t')
plt.show()
'''
