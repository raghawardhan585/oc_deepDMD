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
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import itertools
import copy
from scipy.stats import pearsonr as corr
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
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 91
NO_OF_ITERATIONS_PER_GPU = 2
NO_OF_ITERATIONS_IN_CPU = 2

dict_hp={}
dict_hp['x']={}
dict_hp['x']['ls_dict_size'] = [3,3,3,4,4]
dict_hp['x']['ls_nn_layers'] = [8,9]
dict_hp['x']['ls_nn_nodes'] = [6,8]
dict_hp['y']={}
dict_hp['y']['ls_dict_size'] = [1,1,1]
dict_hp['y']['ls_nn_layers'] = [7,8,9]
dict_hp['y']['ls_nn_nodes'] = [2,4]
dict_hp['xy']={}
dict_hp['xy']['ls_dict_size'] = [1,1,1,2,2,2,3,3]
dict_hp['xy']['ls_nn_layers'] = [8,9]
dict_hp['xy']['ls_nn_nodes'] = [3,4]
process_variable = 'x'
SYSTEM_NO = DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR

ls_dict_size = dict_hp[process_variable]['ls_dict_size']
ls_nn_layers = dict_hp[process_variable]['ls_nn_layers']
ls_nn_nodes = dict_hp[process_variable]['ls_nn_nodes']
a = list(itertools.product(ls_dict_size,ls_nn_layers,ls_nn_nodes))
print('[INFO] TOTAL NUMBER OF RUNS SCHEDULED : ',len(a))
dict_all_run_conditions ={}
for i in range(len(a)):
    dict_all_run_conditions[i] ={}
    for items in ['x','y','xy']:
        if items != process_variable:
            dict_all_run_conditions[i][items] = {'dict_size': 1, 'nn_layers': 1,'nn_nodes': 1}
        else:
            dict_all_run_conditions[i][process_variable] = {'dict_size': a[i][0], 'nn_layers': a[i][1],'nn_nodes': a[i][2]}
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
    items.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] \n')

ls_run_no =[0,0,0]
for i in dict_all_run_conditions.keys():
    run_params = ''
    for items in dict_all_run_conditions[i].keys():
        for sub_items in dict_all_run_conditions[i][items].keys():
            run_params = run_params + ' ' + str(dict_all_run_conditions[i][items][sub_items])
    if np.mod(i,10) ==0 or np.mod(i,10) ==1: # Microtensor CPU 0
        general_run = 'python3 ocdeepDMD_Sequential.py \'/cpu:0\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[0]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[0]) + '.txt &\n'
        ls_files[0].write(general_run + run_params + write_to_file)
        ls_files[0].write('wait \n')
        ls_run_no[0] = ls_run_no[0] + 1
    elif np.mod(i,10)==9: # Goldentensor GPU 3
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:3\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_files[1].write('wait \n')
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 8: # Goldentensor GPU 2
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:2\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 7: # Goldentensor GPU 1
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:1\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10)==6: # Goldentensor GPU 0
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:0\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 5: # Optictensor GPU 3
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:3\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + run_params + write_to_file)
        ls_files[2].write('wait \n')
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10) == 4: # Optictensor GPU 2
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:2\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + run_params + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10)==3: # Optictensor GPU 1
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:1\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + run_params + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10) == 2: # Optictensor GPU 0
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:0\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + run_params + write_to_file)
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

## Transfer the oc deepDMD files

seq.transfer_current_ocDeepDMD_run_files()

## RUN 1 PROCESSING - Generate predictions and error
# SYSTEM_NO = 10
# ls_process_runs = list(range(0,45)) # Runs for which we want to calculate the error
# SYSTEM_NO = 11
# ls_process_runs = list(range(0,30)) # Runs for which we want to calculate the error
# SYSTEM_NO = 30
# ls_process_runs = list(range(52,62)) # Runs for which we want to calculate the error
# SYSTEM_NO = 53
# ls_process_runs = list(range(0,283)) # Runs for which we want to calculate the error
# SYSTEM_NO = 60
# ls_process_runs = list(range(0,41)) # Runs for which we want to calculate the error
# ls_process_runs = list(range(84,85))
# SYSTEM_NO = 61
# ls_process_runs = list(range(0,18))
# ls_process_runs = list(range(84,85))
# SYSTEM_NO = 70
# ls_process_runs = list(range(0,80)) # Runs for which we want to calculate the error
# SYSTEM_NO = 80
# ls_process_runs = list(range(0,106)) # Runs for which we want to calculate the error
SYSTEM_NO = 90
ls_process_runs = list(range(0,68)) # Runs for which we want to calculate the error
seq.generate_predictions_pickle_file(SYSTEM_NO,state_only =True,ls_process_runs=ls_process_runs)
seq.generate_df_error(SYSTEM_NO,ls_process_runs)


## RUN 1 - Display hyperparameters of the runs
# SYSTEM_NO = 11
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# ls_process_runs = list(range(0,88))
for run in ls_process_runs:
    with open(sys_folder_name + '/Sequential/RUN_' + str(run) + '/dict_hyperparameters.pickle', 'rb') as handle:
        dict_hp = pickle.load(handle)
    print('RUN : ',run,' x_obs: ',dict_hp['x_obs'],' x_layers ',dict_hp['x_layers'],' x_nodes ',dict_hp['x_nodes'],' y_obs: ',dict_hp['y_obs'],' y_layers ',dict_hp['y_layers'],' y_nodes ',dict_hp['y_nodes'],' xy_obs: ',dict_hp['xy_obs'],' xy_layers ',dict_hp['xy_layers'],' xy_nodes ',dict_hp['xy_nodes'])
    # print('RUN : ', run, ' x_obs: ', dict_hp['x_obs'], ' x_layers ', dict_hp['x_layers'], ' x_nodes ', dict_hp['x_nodes'])


## RUN 1 PROCESSING - Get the optimal run from the specified runs
# SYSTEM_NO = 22
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/df_error_SEQUENTIAL.pickle','rb') as handle:
    df_error = pickle.load(handle)
df_error_const_obs = df_error
df_training_plus_validation = df_error_const_obs.train + df_error_const_obs.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index)[0])
# opt_run = 1
# opt_run = 13 #13,18,19
dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)


print('Optimal Run no: ',opt_run)
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)
df_opt_stat = seq.get_run_performance_stats(SYSTEM_NO,opt_run)
##
# Plotting the fit of the required indices
# SYSTEM_NO = 5
# opt_run = 1
# dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)
plot_params ={}
plot_params['individual_fig_height'] = 4 #2
plot_params['individual_fig_width'] = 4#2.4
with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle','rb') as handle:
    var_i = pickle.load(handle)
N_CURVES = len(var_i.keys())
del var_i
ls_train_curves = list(range(int(np.floor(N_CURVES/3))))
ls_valid_curves = list(range(ls_train_curves[-1] + 1 ,ls_train_curves[-1] + 1 + int(np.floor(N_CURVES/3))))
ls_test_curves = list(range(ls_valid_curves[-1]+1,N_CURVES))
f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves[0:20],scaled=False,observables=True,one_step=False)
# f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves[0:20],scaled=True,observables=True,one_step=True)
# f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_test_curves[0:20],scaled=False,observables=True,one_step=False)

## RUN 1 PROCESSING - Display the hyper parameters
# opt_run = 25 # Use this only if we want to see the hyperparameters of a specific run
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)
#========================================================================================================================

## RUN 2 - Training error plot [USELESS UNLESS DEBUGGING]
# SYSTEM_NO = 10
# ls_run_no = list(range(45,63))
# SYSTEM_NO = 11
# ls_run_no = list(range(30,60)) #48
# SYSTEM_NO = 53
# ls_run_no = list(range(284,316))
# SYSTEM_NO = 60
# ls_run_no = list(range(42,60))
SYSTEM_NO = 61
ls_run_no = list(range(18,32))
# SYSTEM_NO = 70
# ls_run_no = list(range(80,110))
# SYSTEM_NO = 80
# ls_run_no = list(range(0,0))
plot_params ={}
plot_params['xy_label_font_size']=9
plot_params['individual_fig_width']=2
plot_params['individual_fig_height']=2
seq.plot_training_runs_output(SYSTEM_NO,ls_run_no,plot_params)
## RUN 2 - Caluculate error and find the optimal run
# SYSTEM_NO = 31
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle','rb') as handle:
    var_i = pickle.load(handle)
N_CURVES = len(var_i.keys())
del var_i
ls_train_curves = list(range(int(np.floor(N_CURVES/3))))
ls_valid_curves = list(range(ls_train_curves[-1] + 1 ,ls_train_curves[-1] + 1 + int(np.floor(N_CURVES/3))))
ls_test_curves = list(range(ls_valid_curves[-1]+1,N_CURVES))

ls_process_run_indices = ls_run_no#list(range(45,72))
seq.generate_predictions_pickle_file_output_only(SYSTEM_NO,ls_process_run_indices)
seq.generate_df_error_output(SYSTEM_NO)
with open(sys_folder_name + '/df_error_SEQUENTIAL_OUTPUT.pickle','rb') as handle:
    df_error = pickle.load(handle)
df_training_plus_validation = df_error.train + df_error.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))
dict_predictions_opt_run = seq.get_prediction_data_output(SYSTEM_NO,opt_run)
plot_params ={}
plot_params['individual_fig_height'] = 5 #2
plot_params['individual_fig_width'] = 4#2.4
print('Optimal Run no: ',opt_run)
f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=False)
f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False)
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)
# ## RUN 2 - Get info on any other run that we might deem worthy
# run_no = 56
#
# dict_predictions_opt_run = seq.get_prediction_data_output(SYSTEM_NO,run_no)
# plot_params ={}
# plot_params['individual_fig_height'] = 5 #2
# plot_params['individual_fig_width'] = 4#2.4
# print('Run no: ',run_no)
# # f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=False)
# f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False)
# #
# with open(sys_folder_name + '/Sequential/RUN_' + str(run_no) + '/dict_hyperparameters.pickle','rb') as handle:
#     dict_hp = pickle.load(handle)
# print(dict_hp)

## ------------------------------------------------------------------------------------------------------------------------------


# Final Runs
# SYSTEM_NO = 10
# ls_process_runs = list(range(63,81))
# SYSTEM_NO = 11
# ls_process_runs = list(range(60,87))
# SYSTEM_NO = 53
# ls_process_runs = list(range(316,348))
# SYSTEM_NO = 60
# ls_process_runs = list(range(60,78))
SYSTEM_NO = 61
ls_process_runs = list(range(32,64))
# SYSTEM_NO = 70
# ls_process_runs = list(range(110,170))
# SYSTEM_NO = 80
# ls_process_runs = list(range(0,0))
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
seq.generate_predictions_pickle_file(SYSTEM_NO,state_only =False,ls_process_runs=ls_process_runs)
seq.generate_df_error_x_and_y(SYSTEM_NO,ls_process_runs)
with open(sys_folder_name + '/df_error_SEQUENTIAL_x_and_y.pickle','rb') as handle:
    df_error = pickle.load(handle)
df_training_plus_validation = df_error.train + df_error.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))
print('Optimal Run : ', opt_run)
# opt_run = 67
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print('Optimal Run Hyperparameters')
print(dict_hp)
df_opt_stat = seq.get_run_performance_stats(SYSTEM_NO,opt_run)
dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)


##


plot_params ={}
plot_params['individual_fig_height'] = 5 #2
plot_params['individual_fig_width'] = 4#2.4

with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle','rb') as handle:
    var_i = pickle.load(handle)
N_CURVES = len(var_i.keys())
del var_i
ls_train_curves = list(range(int(np.floor(N_CURVES/3))))
ls_valid_curves = list(range(ls_train_curves[-1] + 1 ,ls_train_curves[-1] + 1 + int(np.floor(N_CURVES/3))))
ls_test_curves = list(range(ls_valid_curves[-1]+1,N_CURVES))
# f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=True,observables=True,one_step = True)
f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False,observables=True,one_step=True)

## Plotting the observables
plot_params={}
plot_params['xy_label_font_size']=5
plot_params['individual_fig_width']=5
plot_params['individual_fig_height']=5
f2 = seq.plot_observables(dict_predictions_opt_run,plot_params)

## RUN 1 - Saving the Optimal first run result
# SYSTEM_NO = 10
# RUN_NO = 0
# SYSTEM_NO = 11
# RUN_NO = 5
# SYSTEM_NO = 53# Run no 115 for system 23
# RUN_NO = 234
# SYSTEM_NO = 60
# RUN_NO = 19
# SYSTEM_NO = 61
# RUN_NO = 2
# SYSTEM_NO = 70
# RUN_NO = 56
# SYSTEM_NO = 80
# RUN_NO = 99
SYSTEM_NO = 90
RUN_NO = 40
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    d = pickle.load(handle)
with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
    d1 = pickle.load(handle)
for items in d1.keys():
    d[items] = d1[items]
print(d.keys())
with open('/Users/shara/Desktop/oc_deepDMD/System_'+str(SYSTEM_NO)+'_BestRun_1.pickle','wb') as handle:
    pickle.dump(d,handle)

## RUN 2 - Saving the Optimal Results of the Second Run
# SYSTEM_NO = 10
# RUN_NO = 45
# SYSTEM_NO = 11
# RUN_NO = 57#32
# SYSTEM_NO = 53
# RUN_NO = 308
# SYSTEM_NO = 60
# RUN_NO = 58
SYSTEM_NO = 61
RUN_NO = 18 #27
# SYSTEM_NO = 70
# RUN_NO = 104
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    d = pickle.load(handle)
with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
    d1 = pickle.load(handle)
for items in d1.keys():
    d[items] = d1[items]
print(d.keys())
with open('/Users/shara/Desktop/oc_deepDMD/System_'+str(SYSTEM_NO)+'_BestRun_2.pickle','wb') as handle:
    pickle.dump(d,handle)


## RUN 1 PROCESSING - Filtering the runs based on the hyperparameters
ls_process_runs = list(range(0,92))
ls_filtered_runs =[]
SYSTEM_NO = 23
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
for run_i in ls_process_runs:
    with open(sys_folder_name + '/Sequential/RUN_' + str(run_i) + '/dict_hyperparameters.pickle', 'rb') as handle:
        dict_hp_i = pickle.load(handle)
    if dict_hp_i['x_obs'] == 9:
        ls_filtered_runs.append(run_i)
print(ls_filtered_runs)

## Calculating the required error statistics [TO PUT IN THE PAPER]

SYS_NO = 53
RUN_NO = 344

sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYS_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
    d = pickle.load(handle)[RUN_NO]
if 'observables' in d.keys():
    N_CURVES = len(d.keys()) - 4
else:
    N_CURVES = len(d.keys())
ls_train_curves = list(range(int(np.floor(N_CURVES / 3))))
ls_valid_curves = list(range(ls_train_curves[-1] + 1, ls_train_curves[-1] + 1 + int(np.floor(N_CURVES / 3))))
ls_test_curves = list(range(ls_valid_curves[-1] + 1, N_CURVES))

dict_error = {'train':{},'valid':{},'test':{}}
for items in dict_error.keys():
    dict_error[items]['r2'] = []
    dict_error[items]['r2_n'] = []
for i in ls_train_curves:
    SST = np.sum(np.square(d[i]['psiX']))
    SSE = np.sum(np.square(d[i]['psiX'] - d[i]['psiX_est_one_step']))
    SSE_n = np.sum(np.square(d[i]['psiX'] - d[i]['psiX_est_n_step']))
    dict_error['train']['r2'].append((1 - SSE/SST)*100)
    dict_error['train']['r2_n'].append((1 - SSE_n / SST) * 100)
for i in ls_valid_curves:
    SST = np.sum(np.square(d[i]['psiX']))
    SSE = np.sum(np.square(d[i]['psiX'] - d[i]['psiX_est_one_step']))
    SSE_n = np.sum(np.square(d[i]['psiX'] - d[i]['psiX_est_n_step']))
    dict_error['valid']['r2'].append((1 - SSE/SST)*100)
    dict_error['valid']['r2_n'].append((1 - SSE_n / SST) * 100)
for i in ls_test_curves:
    SST = np.sum(np.square(d[i]['psiX']))
    SSE = np.sum(np.square(d[i]['psiX'] - d[i]['psiX_est_one_step']))
    SSE_n = np.sum(np.square(d[i]['psiX'] - d[i]['psiX_est_n_step']))
    dict_error['test']['r2'].append((1 - SSE/SST)*100)
    dict_error['test']['r2_n'].append((1 - SSE_n / SST) * 100)

print('2 - step Training r2 accuracy: ', np.mean(dict_error['train']['r2'] ))
print('1 - step Validation r2 accuracy: ', np.mean(dict_error['valid']['r2'] ))
print('1 - step Testing r2 accuracy: ', np.mean(dict_error['test']['r2'] ))

print('N - step Training r2 accuracy: ', np.mean(dict_error['train']['r2_n'] ))
print('N - step Validation r2 accuracy: ', np.mean(dict_error['valid']['r2_n'] ))
print('N - step Testing r2 accuracy: ', np.mean(dict_error['test']['r2_n'] ))

##
SYSTEM_NO = 80
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle','rb') as handle:
    d = pickle.load(handle)
##
RUN_NO = 99
plt.figure()
for i in range(300):
    plt.plot(d[RUN_NO][i]['X_est_n_step'][:,0])
plt.show()


## SPECIFIC TO SYSTEM 11 - Comparing the computed output with the
# SYSTEM_NO = 11
# ls_output_runs = list(range(30,60))
SYSTEM_NO = 61
ls_output_runs = list(range(18,32))
NORMALIZE = True

x1 = np.arange(-10, 10.5, 0.5)
x2 = np.arange(-10, 10.5, 0.5)
# x2 = np.arange(-150, 20, 4)
X1, X2 = np.meshgrid(x1, x2)
Y_theo = np.zeros(shape=(X1.shape[0], X1.shape[1],1))

if SYSTEM_NO in [10,11]:
    for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
        Y_theo[i,j,0] = X1[i, j]*X2[i, j]
elif SYSTEM_NO in [61]:
    for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
        Y_theo[i,j,0] = X2[i, j] / (1 + 0.6 * X1[i, j] ** 3)

N_RUNS = len(ls_output_runs)
N_COLS = np.int(np.ceil(np.sqrt(N_RUNS+1)))
N_ROWS = np.int(np.ceil((N_RUNS+1)/N_COLS))

f,ax = plt.subplots(N_ROWS,N_COLS,sharex=True,sharey=True,figsize=(3*N_COLS,3*N_ROWS))
ax = ax.reshape(-1)
if NORMALIZE:
    c = ax[0].pcolor(X1, X2, Y_theo[:,:,0] / np.max(np.abs(Y_theo[:,:,0])), cmap='rainbow', vmin=-1, vmax=1)
else:
    c = ax[0].pcolor(X1, X2, Y_theo[:,:,0], cmap='rainbow', vmin=np.min(Y_theo[:,:,0]), vmax=np.max(Y_theo))
p = 1
Y_all = copy.deepcopy(Y_theo)
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
for run_i in ls_output_runs:
    run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(run_i)
    if os.path.exists(run_folder_name):
        print('RUN: ', run_i)
        sess = tf.InteractiveSession()
        dict_params, _, dict_indexed_data = seq.get_all_run_info_output(SYSTEM_NO, run_i, sess)
        Y_now = np.zeros(shape=(X1.shape[0], X1.shape[1], 1))
        for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
            Y_now[i,j,0] = dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: np.array([[X1[i, j], X2[i, j]]])})[0][-1]
        if NORMALIZE:
            c = ax[p].pcolor(X1, X2, Y_now[:, :, 0] / np.max(np.abs(Y_now[:, :, 0])), cmap='rainbow', vmin=-1, vmax=1)
        else:
            c = ax[p].pcolor(X1, X2, Y_now[:, :, 0], cmap='rainbow', vmin=np.min(Y_now[:, :, 0]), vmax=np.max(Y_now[:, :, 0]))
        Y_all = np.concatenate([Y_all,Y_now],axis=2)
        if np.mod(p+1,N_COLS) == 0:
            f.colorbar(c, ax=ax[p])
        ax[p].set_xticks([-8, 0, 8])
        ax[p].set_yticks([-8, 0, 8])
        ax[p].set_title('Run- ' + str(run_i) + ' corr - ' + str(round(corr(Y_theo.reshape(-1),Y_now.reshape(-1))[0],2)))
        tf.reset_default_graph()
        sess.close()
        p = p + 1
f.show()

##

N_STEPS = 1000
SYSTEM_NO = 61
RUN_NO = 2
# sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
sess = tf.InteractiveSession()
dict_params, _, dict_indexed_data = seq.get_all_run_info(SYSTEM_NO, RUN_NO, sess)
f2,ax = plt.subplots(3,1)
x_i_range = np.arange(-2,-1.5,0.5)
for x0_1,x0_2 in itertools.product(x_i_range,x_i_range):
    X = np.array([[x0_1,x0_2]])
    psiXT = dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: X[-1:]})
    for i in range(N_STEPS):
        psiXT = np.matmul(psiXT,dict_params['KxT_num'])
        X = np.concatenate([X,psiXT[:,0:2]],axis=0)
    ax[0].plot(X[:,0],X[:,1],'b')
    ax[1].plot(X[:,0],'b')
    ax[2].plot(X[:,1],'b')
f2.show()
tf.reset_default_graph()
sess.close()


