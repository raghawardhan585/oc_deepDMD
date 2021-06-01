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
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 400
NO_OF_ITERATIONS_PER_GPU = 2
NO_OF_ITERATIONS_IN_CPU = 2

dict_hp={}
dict_hp['x']={}
dict_hp['x']['ls_dict_size'] = [5]
dict_hp['x']['ls_nn_layers'] = [3]
dict_hp['x']['ls_nn_nodes'] = [10]
dict_hp['y']={}
dict_hp['y']['ls_dict_size'] = [1,1]
dict_hp['y']['ls_nn_layers'] = [8,9]
dict_hp['y']['ls_nn_nodes'] = [4,6,8]
dict_hp['xy']={}
dict_hp['xy']['ls_dict_size'] = [2,3,4]
dict_hp['xy']['ls_nn_layers'] = [8,9]
dict_hp['xy']['ls_nn_nodes'] = [6,8]
process_variable = 'x'
SYSTEM_NO = DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR

ls_dict_size = dict_hp[process_variable]['ls_dict_size']
ls_nn_layers = dict_hp[process_variable]['ls_nn_layers']
ls_nn_nodes = dict_hp[process_variable]['ls_nn_nodes']
ls_regularization_parameter = np.arange(0, 1e-3, 2.5e-5) #[3.75e-4] #np.arange(5e-5,1e-3,2.5e-5)
# a = list(itertools.product(ls_dict_size,ls_nn_layers,ls_nn_nodes))
a = list(itertools.product(ls_dict_size,ls_nn_layers,ls_nn_nodes,ls_regularization_parameter))
for i in range(len(a)):
    if a[i][0] ==0:
        # a[i] = (0,1,0)
        a[i] = (0, 1, 0, a[i][-1])

print('[INFO] TOTAL NUMBER OF RUNS SCHEDULED : ',len(a))
dict_all_run_conditions ={}
for i in range(len(a)):
    dict_all_run_conditions[i] ={}
    for items in ['x','y','xy']:
        if items != process_variable:
            dict_all_run_conditions[i][items] = {'dict_size': 1, 'nn_layers': 1,'nn_nodes': 1}
        else:
            dict_all_run_conditions[i][process_variable] = {'dict_size': a[i][0], 'nn_layers': a[i][1],'nn_nodes': a[i][2]}
    dict_all_run_conditions[i]['regularization lambda'] = a[i][-1]
print(dict_all_run_conditions)
# Scheduling
mt = open('/Users/shara/Desktop/oc_deepDMD/microtensor_run.sh','w')
gt = open('/Users/shara/Desktop/oc_deepDMD/goldentensor_run.sh','w')
ot = open('/Users/shara/Desktop/oc_deepDMD/optictensor_run.sh','w')
ls_files  = [mt,gt,ot]
# ls_files  = [gt]
for items in ls_files:
    items.write('#!/bin/bash \n')
    items.write('rm -rf _current_run_saved_files \n')
    items.write('mkdir _current_run_saved_files \n')
    items.write('rm -rf Run_info \n')
    items.write('mkdir Run_info \n')
    items.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] \n')

ls_run_no =[0,0]
for i in dict_all_run_conditions.keys():
    run_params = ''
    for items in ['x','y','xy']:
        for sub_items in dict_all_run_conditions[i][items].keys():
            run_params = run_params + ' ' + str(dict_all_run_conditions[i][items][sub_items])
    run_params = run_params + ' ' + str(dict_all_run_conditions[i]['regularization lambda'])
    # if np.mod(i,10) ==0 or np.mod(i,10) ==1: # Microtensor CPU 0
    #     general_run = 'python3 ocdeepDMD_Sequential.py \'/cpu:0\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[0]) + ' '
    #     write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[0]) + '.txt &\n'
    #     ls_files[0].write(general_run + run_params + write_to_file)
    #     ls_files[0].write('wait \n')
    #     ls_run_no[0] = ls_run_no[0] + 1
    if np.mod(i,4) ==3: # Goldentensor GPU 3
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:3\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_files[1].write('wait \n')
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,4) == 2: # Goldentensor GPU 2
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:2\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,4) == 1: # Goldentensor GPU 1
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:1\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,4)==0: # Goldentensor GPU 0
        general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:0\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + run_params + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    # elif np.mod(i,10) == 5: # Optictensor GPU 3
    #     general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:3\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
    #     write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
    #     ls_files[2].write(general_run + run_params + write_to_file)
    #     ls_files[2].write('wait \n')
    #     ls_run_no[2] = ls_run_no[2] + 1
    # elif np.mod(i,10) == 4: # Optictensor GPU 2
    #     general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:2\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
    #     write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
    #     ls_files[2].write(general_run + run_params + write_to_file)
    #     ls_run_no[2] = ls_run_no[2] + 1
    # elif np.mod(i,10)==3: # Optictensor GPU 1
    #     general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:1\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
    #     write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
    #     ls_files[2].write(general_run + run_params + write_to_file)
    #     ls_run_no[2] = ls_run_no[2] + 1
    # elif np.mod(i,10) == 2: # Optictensor GPU 0
    #     general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:0\' ' + str(SYSTEM_NO) + ' ' + str(ls_run_no[2]) + ' '
    #     write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
    #     ls_files[2].write(general_run + run_params + write_to_file)
    #     ls_run_no[2] = ls_run_no[2] + 1
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


##

