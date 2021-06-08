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
import deepDMD_helper_functions as dp
import ocdeepdmd_simulation_examples_helper_functions as oc
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
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 502#406
NO_OF_ITERATIONS_PER_GPU = 1
NO_OF_ITERATIONS_IN_CPU = 1

dict_hp={}
dict_hp['ls_dict_size'] = [4,5,6,7]
dict_hp['ls_nn_layers'] = [3,4]
dict_hp['ls_nn_nodes'] = [20]
SYSTEM_NO = DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR

ls_dict_size = dict_hp['ls_dict_size']
ls_nn_layers = dict_hp['ls_nn_layers']
ls_nn_nodes = dict_hp['ls_nn_nodes']
a = list(itertools.product(ls_dict_size,ls_nn_layers,ls_nn_nodes))
print('[INFO] TOTAL NUMBER OF RUNS SCHEDULED : ',len(a))
dict_all_run_conditions ={}
for i in range(len(a)):
    dict_all_run_conditions[i] = str(a[i][0]) + ' ' + str(a[i][1]) + ' ' + str(a[i][2])
print(dict_all_run_conditions)
# Scheduling
ot = open('/Users/shara/Desktop/oc_deepDMD/optictensor_run.sh','w')

ot.write('#!/bin/bash \n')
ot.write('rm -rf _current_run_saved_files \n')
ot.write('mkdir _current_run_saved_files \n')
ot.write('rm -rf Run_info \n')
ot.write('mkdir Run_info \n')
ot.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] \n')
running_code = 'deepDMD.py'
run_no = 0
for i in dict_all_run_conditions.keys():
    if np.mod(i,4)==3: # Optictensor GPU 3
        general_run = 'python3 ' + running_code + ' \'/gpu:3\' ' + str(SYSTEM_NO) + ' ' + str(run_no) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(run_no) + '.txt &\n'
        ot.write(general_run + dict_all_run_conditions[i] + write_to_file)
        ot.write('wait \n')
        run_no = run_no + 1
    elif np.mod(i,4) == 2: # Optictensor GPU 2
        general_run = 'python3 ' + running_code + ' \'/gpu:2\' ' + str(SYSTEM_NO) + ' ' + str(run_no) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(run_no) + '.txt &\n'
        ot.write(general_run + dict_all_run_conditions[i] + write_to_file)
        run_no = run_no + 1
    elif np.mod(i,4) == 1: # Optictensor GPU 1
        general_run = 'python3 ' + running_code + ' \'/gpu:1\' ' + str(SYSTEM_NO) + ' ' + str(run_no) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(run_no) + '.txt &\n'
        ot.write(general_run + dict_all_run_conditions[i] + write_to_file)
        run_no = run_no + 1
    elif np.mod(i,4)==0: # Optictensor GPU 0
        general_run = 'python3 ' + running_code + ' \'/gpu:0\' ' + str(SYSTEM_NO) + ' ' + str(run_no) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(run_no) + '.txt &\n'
        ot.write(general_run + dict_all_run_conditions[i] + write_to_file)
        run_no = run_no + 1

ot.write('wait \n')
ot.write('echo "All sessions are complete" \n')
ot.write('echo "=======================================================" \n')
ot.write('cd .. \n')
ot.write('rm -R _current_run_saved_files \n')
ot.write('rm -R Run_info \n')
ot.write('cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files \n')
ot.write('cp -a oc_deepDMD/Run_info/ Run_info \n')
ot.write('cd oc_deepDMD/ \n')
ot.close()
##
import deepDMD_helper_functions as dp
dp.transfer_current_ocDeepDMD_run_files()