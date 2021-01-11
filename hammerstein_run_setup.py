##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pickle
import random
import os
import shutil
import tensorflow as tf
import Sequential_Helper_Functions as seq
import hammerstein_helper_functions as hm
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools

import ocdeepdmd_simulation_examples_helper_functions as oc
colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


## Bash Script Generation
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 10
NO_OF_ITERATIONS_PER_GPU = 2
NO_OF_ITERATIONS_IN_CPU = 2

dict_hp={}
dict_hp['x']={}
dict_hp['x']['ls_nn_layers'] = [3,4,5]
dict_hp['x']['ls_nn_nodes'] = [5,10,15]
dict_hp['y']={}
dict_hp['y']['ls_nn_layers'] = [3,4,5]
dict_hp['y']['ls_nn_nodes'] = [5,10,15]
process_variable = 'y'
SYSTEM_NO = DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR

ls_nn_layers = dict_hp[process_variable]['ls_nn_layers']
ls_nn_nodes = dict_hp[process_variable]['ls_nn_nodes']
a = list(itertools.product(ls_nn_layers,ls_nn_nodes))
print('[INFO] TOTAL NUMBER OF RUNS SCHEDULED : ',len(a))
dict_all_run_conditions ={}
for i in range(len(a)):
    dict_all_run_conditions[i] = str(a[i][0]) + ' '  + str(a[i][1])
print(dict_all_run_conditions)
# Scheduling
mt = open('/Users/shara/Desktop/oc_deepDMD/microtensor_run.sh','w')
gt = open('/Users/shara/Desktop/oc_deepDMD/goldentensor_run.sh','w')
ot = open('/Users/shara/Desktop/oc_deepDMD/optictensor_run.sh','w')
ls_files  = [mt,gt,ot]
for items in ls_files:
    items.write('#!/bin/bash \n')
    items.write('rm -rf _current_run_saved_files \n')
    items.write('mkdir _current_run_saved_files \n')
    items.write('rm -rf Run_info \n')
    items.write('mkdir Run_info \n')
    items.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [process var] [run_no] [n_layers] [n_nodes] [write_to_file] \n')
running_code = 'hammerstein_nn_identification.py'
ls_run_no =[0,0,0]
for i in dict_all_run_conditions.keys():
    if np.mod(i,10) ==0 or np.mod(i,10) ==1: # Microtensor CPU 0
        general_run = 'python3 ' + running_code + ' \'/cpu:0\' ' + str(SYSTEM_NO) + ' \'' + process_variable + '\' ' + str(ls_run_no[0]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[0]) + '.txt &\n'
        ls_files[0].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_files[0].write('wait \n')
        ls_run_no[0] = ls_run_no[0] + 1
    elif np.mod(i,10)==9: # Goldentensor GPU 3
        general_run = 'python3 ' + running_code + ' \'/gpu:3\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_files[1].write('wait \n')
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 8: # Goldentensor GPU 2
        general_run = 'python3 ' + running_code + ' \'/gpu:2\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 7: # Goldentensor GPU 1
        general_run = 'python3 ' + running_code + ' \'/gpu:1\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10)==6: # Goldentensor GPU 0
        general_run = 'python3 ' + running_code + ' \'/gpu:0\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 5: # Optictensor GPU 3
        general_run = 'python3 ' + running_code + ' \'/gpu:3\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_files[2].write('wait \n')
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10) == 4: # Optictensor GPU 2
        general_run = 'python3 ' + running_code + ' \'/gpu:2\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10)==3: # Optictensor GPU 1
        general_run = 'python3 ' + running_code + ' \'/gpu:1\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10) == 2: # Optictensor GPU 0
        general_run = 'python3 ' + running_code + ' \'/gpu:0\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
for items in ls_files:
    items.write('wait \n')
    items.write('echo "All sessions are complete" \n')
    items.write('echo "=======================================================" \n')
    items.write('cd .. \n')
    items.write('rm -R _current_run_saved_files \n')
    items.write('rm -R Run_info \n')
    items.write('cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files \n')
    items.write('cp -a oc_deepDMD/Run_info/ Run_info \n')
    items.write('cd oc_deepDMD/ \n')
    items.close()

##
hm.transfer_current_ocDeepDMD_run_files()
## RUN 1 PROCESSING - Generate predictions and error
# SYSTEM_NO = 110
# ls_process_runs = list(range(0,45))
# SYSTEM_NO = 130
# ls_process_runs = list(range(52,62))
# SYSTEM_NO = 153
# ls_process_runs = list(range(0,283))
SYSTEM_NO = 10
ls_process_runs = list(range(0,5))

sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# Make a predictions folder if one doesn't exist
if os.path.exists(sys_folder_name + '/dict_predictions_HAMMERSTEIN.pickle'):
    with open(sys_folder_name + '/dict_predictions_HAMMERSTEIN.pickle','rb') as handle:
        dict_predictions_SEQUENTIAL = pickle.load(handle)
else:
    dict_predictions_SEQUENTIAL = {}
# Find all available run folders
ls_all_run_indices = []
for folder in os.listdir(sys_folder_name+'/Hammerstein'):
    if folder[0:4] == 'RUN_': # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
# List of all processed runs are the keys of dict_prediction_SEQUENTIAL
ls_processed_runs = list(dict_predictions_SEQUENTIAL.keys())
ls_unprocessed_runs = list(set(ls_all_run_indices) - set(ls_processed_runs))
# Among the unprocessed runs, only process the specified runs
if len(ls_process_runs) !=0:
    ls_unprocessed_runs = list(set(ls_unprocessed_runs).intersection(set(ls_process_runs)))
print('RUNS TO PROCESS - ',ls_unprocessed_runs)



for run in ls_unprocessed_runs:
    print('RUN: ', run)
    run_folder_name = sys_folder_name + '/Hammerstein/RUN_' + str(run)
    with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
        d = pickle.load(handle)
    print(d)
##

dict_predictions_HAMMERSTEIN = {}
for run in ls_unprocessed_runs:
    run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(run)
    with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
        d = pickle.load(handle)

    dict_predictions_HAMMERSTEIN[run] = {}
    sess = tf.InteractiveSession()
    dict_params, _, dict_indexed_data = seq.get_all_run_info(SYSTEM_NO, run, sess)
    if d['process_variable'] == 'x':
        # Get the 1-step and n-step prediction data
    elif d['process_variable'] == 'y':
        # Get the output data fit
        try: # If there exists an OPTIMAL_STATE_FIT





    if state_only:
        dict_intermediate = oc.model_prediction_state_only(dict_indexed_data, dict_params, SYSTEM_NO)
    else:
        dict_intermediate = oc.model_prediction(dict_indexed_data, dict_params, SYSTEM_NO)
    for curve_no in dict_intermediate.keys():
        dict_predictions_SEQUENTIAL[run][curve_no] = dict_intermediate[curve_no]



    tf.reset_default_graph()
    sess.close()
# Saving the dict_predictions folder
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'wb') as handle:
    pickle.dump(dict_predictions_SEQUENTIAL, handle)




# seq.generate_predictions_pickle_file(SYSTEM_NO,state_only =True,ls_process_runs=ls_process_runs)
# seq.generate_df_error(SYSTEM_NO,ls_process_runs)