##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import ocdeepdmd_simulation_examples_helper_functions as oc

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


SYSTEM_NO = 2


sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# Make a predictions folder
if os.path.exists(sys_folder_name + '/dict_predictions.pickle'):
    with open(sys_folder_name + '/dict_predictions.pickle','rb') as handle:
        dict_predictions = pickle.load(handle)
else:
    dict_predictions = {}


ls_processed_runs = list(dict_predictions.keys())
# Scan all folders to get all Run Indices
ls_all_run_indices =[]
for folder in os.listdir(sys_folder_name):
    if folder[0:4] == 'RUN_': # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
ls_unprocessed_runs = list(set(ls_all_run_indices) - set(ls_processed_runs))
print('RUNS TO PROCESS - ',ls_unprocessed_runs)


for run in ls_unprocessed_runs:
    print('RUN: ', run)
    dict_predictions[run]={}
    sess = tf.InteractiveSession()
    dict_params, _, dict_indexed_data, __, ___ = oc.get_all_run_info(SYSTEM_NO, run, sess)
    sampling_resolution = 0.01
    dict_psi_phi = oc.observables_and_eigenfunctions(dict_params, sampling_resolution)
    dict_predictions[run]['X1'] = dict_psi_phi['X1']
    dict_predictions[run]['X2'] = dict_psi_phi['X2']
    dict_predictions[run]['observables'] = dict_psi_phi['observables']
    dict_predictions[run]['eigenfunctions'] = dict_psi_phi['eigenfunctions']
    dict_intermediate = oc.model_prediction(dict_indexed_data, dict_params, SYSTEM_NO)
    for curve_no in dict_intermediate.keys():
        dict_predictions[run][curve_no] = dict_intermediate[curve_no]
    tf.reset_default_graph()
    sess.close()

with open(sys_folder_name + '/dict_predictions.pickle','wb') as handle:
    pickle.dump(dict_predictions,handle)





