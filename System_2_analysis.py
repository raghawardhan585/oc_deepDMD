##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf


colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette
sess = tf.InteractiveSession()


SYSTEM_NO = 2


sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# Make a predictions folder
if os.path.exists(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_Predictions.pickle'):
    with open('processed_runs.pickle','rb') as handle:
        processed_runs = pickle.load(handle)
else:
    os.mkdir(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_Predictions')
    processed_runs = []

# Scan all folders to get all Run Indices
ls_all_run_indices =[]
for folder in os.listdir(sys_folder_name):
    if folder[0:4] == 'RUN_': # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
ls_unprocessed_runs = list(set(ls_all_run_indices) - set(processed_runs))



