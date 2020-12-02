##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


SYSTEM_NO = 7
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)


with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'rb') as handle:
    dict_data = pickle.load(handle)

plt.figure()
alpha = 0.5
epsilon = alpha - 0.1
arrow_length = 0.01
ls_pts = list(range(0,20,2))
for i in list(dict_data.keys())[0:200]:
    for j in ls_pts:
        if np.abs(dict_data[i]['X'][j, 0]) > 1 or j==0:
            plt.plot(dict_data[i]['X'][j, 0], dict_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=3)
    plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.5)
    if np.mod(i,1)==0:
        for j in ls_pts[0:2]:
            dist = np.sqrt((dict_data[i]['X'][j, 0] - dict_data[i]['X'][j + 1, 0]) ** 2 + (dict_data[i]['X'][j, 1] - dict_data[i]['X'][j + 1, 1]) ** 2)
            if dist < 2:
                break
            x = alpha * dict_data[i]['X'][j, 0] + (1 - alpha) * dict_data[i]['X'][j+1, 0]
            y = alpha * dict_data[i]['X'][j, 1] + (1 - alpha) * dict_data[i]['X'][j+1, 1]
            dx = (epsilon * dict_data[i]['X'][j, 0] + (1 - epsilon) * dict_data[i]['X'][j+1, 0] - x)*arrow_length*dist
            dy = (epsilon * dict_data[i]['X'][j, 1] + (1 - epsilon) * dict_data[i]['X'][j+1, 1] - y)*arrow_length*dist
            plt.arrow(x,y,dx,dy,head_width = 0.3,head_length=3,alpha=0.5,color='tab:green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot([0],[0],'o',color='tab:red',markersize=10)
plt.show()

plt.figure()


##

