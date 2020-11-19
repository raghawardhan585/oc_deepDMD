##

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
import random
import os
import shutil
import tensorflow as tf
import copy
import itertools
from scipy.integrate import odeint

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette

def write_bash_script(DEVICE_TO_RUN_ON,dict_run_conditions,SYSTEM_NO,NO_OF_ITERATIONS_PER_GPU,NO_OF_ITERATIONS_IN_CPU):
    with open('/Users/shara/Desktop/oc_deepDMD/' + str(DEVICE_TO_RUN_ON) + '_run.sh', 'w') as bash_exec:
        bash_exec.write('#!/bin/bash \n')
        bash_exec.write('rm -rf _current_run_saved_files \n')
        bash_exec.write('mkdir _current_run_saved_files \n')
        bash_exec.write('rm -rf Run_info \n')
        bash_exec.write('mkdir Run_info \n')
        bash_exec.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] \n')
        if DEVICE_TO_RUN_ON in ['optictensor', 'goldentensor']:
            ls_gpu = [0,1,2,3]
            ls_cpu = [4]
        elif DEVICE_TO_RUN_ON == 'microtensor':
            ls_gpu = [-1]
            ls_cpu = [0,1,2,3,4,5,6]
        RUN_NO = 0
        for i in dict_run_conditions.keys():
            if i in ls_gpu:
                for j in range(NO_OF_ITERATIONS_PER_GPU):
                    general_run = 'python3 ocdeepDMD_Sequential.py \'/gpu:' + str(i) + '\' ' + str(SYSTEM_NO) + ' '
                    run_params = str(RUN_NO)
                    for items in dict_run_conditions[i].keys():
                        for sub_items in dict_run_conditions[i][items].keys():
                            run_params = run_params + ' ' + str(dict_run_conditions[i][items][sub_items])
                    write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NO) + '.txt &\n'
                    bash_exec.write(general_run + run_params + write_to_file)
                    RUN_NO = RUN_NO + 1
            elif i in ls_cpu:
                for j in range(NO_OF_ITERATIONS_IN_CPU):
                    general_run = 'python3 ocdeepDMD_Sequential.py \'/cpu:0\' ' + str(SYSTEM_NO) + ' '
                    run_params = str(RUN_NO)
                    for items in dict_run_conditions[i].keys():
                        for sub_items in dict_run_conditions[i][items].keys():
                            run_params = run_params + ' ' + str(dict_run_conditions[i][items][sub_items])
                    write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NO) + '.txt &\n'
                    bash_exec.write(general_run + run_params + write_to_file)
                    RUN_NO = RUN_NO + 1
        bash_exec.write('echo "Running all sessions" \n')
        bash_exec.write('wait \n')
        bash_exec.write('echo "All sessions are complete" \n')
        bash_exec.write('echo "=======================================================" \n')
        bash_exec.write('cd .. \n')
        bash_exec.write('rm -R _current_run_saved_files \n')
        bash_exec.write('rm -R Run_info \n')
        bash_exec.write('cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files \n')
        bash_exec.write('cp -a oc_deepDMD/Run_info/ Run_info \n')
        bash_exec.write('cd oc_deepDMD/ \n')
        # cp -a _current_run_saved_files/ oc_deepDMD/_current_run_saved_files
        # cp -a Run_info/ oc_deepDMD/Run_info
    return

def transfer_current_ocDeepDMD_run_files():
    runinfo_folder = '/Users/shara/Desktop/oc_deepDMD/Run_info'
    source_folder = '/Users/shara/Desktop/oc_deepDMD/_current_run_saved_files'
    # Find the SYSTEM NUMBER
    # Assumption: All the folders in the _current_run_saved_files belong to the same system
    for items in os.listdir(source_folder):
        if items[0:4] == 'SYS_':
            for i in range(4, len(items)):
                if items[i] == '_':
                    SYSTEM_NUMBER = int(items[4:i])
                    break
            break
    # Find HIGHEST RUN NUMBER
    destination_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NUMBER) + '/Sequential'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    current_run_no = -1
    ls_complete_runs = []
    for items in os.listdir(destination_folder):
        if items[0:4] == 'RUN_':
            current_run_no = np.max([current_run_no, int(items[4:])])
            ls_complete_runs.append(int(items[4:]))
    ls_missing_runs = list(set(range(current_run_no))-set(ls_complete_runs))

    # Transfer files to missing folders
    n_miss = len(ls_missing_runs)
    if n_miss>0:
        i = 0
        for items in list(set(os.listdir(source_folder)) - {'.DS_Store'}):
            shutil.move(source_folder + '/' + items, destination_folder + '/RUN_' + str(ls_missing_runs[i]))
            shutil.move(runinfo_folder + '/' + items + '.txt',
                        destination_folder + '/RUN_' + str(ls_missing_runs[i]) + '/RUN_' + str(ls_missing_runs[i]) + '.txt')
            i = i+1
            if i == n_miss:
                break

    # Transfer the files to new folders
    current_run_no = current_run_no + 1
    for items in list(set(os.listdir(source_folder)) - {'.DS_Store'}):
        shutil.move(source_folder + '/' + items, destination_folder + '/RUN_' + str(current_run_no))
        shutil.move(runinfo_folder + '/' + items + '.txt',
                    destination_folder + '/RUN_' + str(current_run_no) + '/RUN_' + str(current_run_no) + '.txt')
        current_run_no = current_run_no + 1
    return