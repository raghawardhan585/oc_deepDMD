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

import ocdeepdmd_simulation_examples_helper_functions as oc

# ==========================
## System 1 Data Generation
# ==========================
# System Parameters
A = np.array([[0.86,0.],[0.8,0.4]])
gamma = 0
# Simulation Parameters
N_data_points = 30
N_CURVES = 10
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
SYSTEM_NO = 1
oc.data_gen_sys_1_2(sys_params, N_CURVES, SYSTEM_NO)

# ==========================
## System 2 Data Generation
# ==========================
# System Parameters
A = np.array([[0.86,0.],[0.8,0.4]])
gamma = -0.4
# Simulation Parameters
N_data_points = 30
N_CURVES = 20
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
SYSTEM_NO = 2
oc.data_gen_sys_1_2(sys_params, N_CURVES, SYSTEM_NO)

# # ==========================
# # System 3 [Yet to be written]
# # ==========================
# # System Parameters
# A = np.array([[0.86,0.],[0.8,0.4]])
# gamma = 0.4
# # Simulation Parameters
# N_data_points = 30
# N_CURVES = 10
# sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
# SYSTEM_NO = 2
# oc.data_gen_sys_1_2(N_data_points, N_CURVES, SYSTEM_NO)

## Bash Script Generator

# DEVICE_TO_RUN_ON = 'microtensor'
# DEVICE_TO_RUN_ON = 'optictensor'
DEVICE_TO_RUN_ON = 'goldentensor'
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 2
NO_OF_ITERATIONS_PER_GPU = 3
NO_OF_ITERATIONS_IN_CPU = 3

dict_run_conditions = {}
# Runs
dict_run_conditions[0] = {'x_dict_size':2,'x_nn_layers':3,'x_nn_nodes':6}
dict_run_conditions[1] = {'x_dict_size':3,'x_nn_layers':3,'x_nn_nodes':6}
dict_run_conditions[2] = {'x_dict_size':4,'x_nn_layers':3,'x_nn_nodes':6}
dict_run_conditions[3] = {'x_dict_size':5,'x_nn_layers':3,'x_nn_nodes':6}
dict_run_conditions[4] = {'x_dict_size':6,'x_nn_layers':3,'x_nn_nodes':6}

oc.write_bash_script(DEVICE_TO_RUN_ON, dict_run_conditions, DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR, NO_OF_ITERATIONS_PER_GPU, NO_OF_ITERATIONS_IN_CPU)

## Transfering the files to the required deepDMD run files
# This is more for organizing all the runs for a single system in a single folder

oc.transfer_current_ocDeepDMD_run_files()

##

SYSTEM_NO = 1
RUN_NO = 1
sess = tf.InteractiveSession()
dict_params,  ls_all_indices, dict_indexed_data,  df_train_learning_curves, df_run_info = oc.get_all_run_info(SYSTEM_NO,RUN_NO,sess)

##
a = oc.model_prediction(dict_indexed_data, dict_params, SYSTEM_NO)
a[0].keys()

## Eigenfunctions and Observables