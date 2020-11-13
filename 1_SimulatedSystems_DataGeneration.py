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
colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette

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
gamma = -0.9
# Simulation Parameters
N_data_points = 30
N_CURVES = 60
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
SYSTEM_NO = 4
oc.data_gen_sys_1_2(sys_params, N_CURVES, SYSTEM_NO)
oc.plot_training_valid_test_states(4)
# ==========================
## System 3 - Activator Repressor CLock - 4state system
# ==========================
# System Parameters
gamma_A = 1.
gamma_B = 0.5
delta_A = 1.
delta_B = 1.
alpha_A0= 0.04
alpha_B0= 0.004
alpha_A = 250.
alpha_B = 30.
K_A = 1.
K_B = 1.
kappa_A = 1.
kappa_B = 1.
n = 2.
m = 4.
k_3n = 3.
k_3d = 1.08
sys_params_arc4s = (gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m)
# Simulation Parameters
sampling_time = 0.1
simulation_time = 30
N_CURVES = 60

sys_params = {'sys_params_arc4s': sys_params_arc4s , 'k_3n':k_3n, 'k_3d':k_3d, 'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES}
SYSTEM_NO = 3
oc.data_gen_sys_arc4s(sys_params, N_CURVES,SYSTEM_NO)

## Bash Script Generator

# DEVICE_TO_RUN_ON = 'microtensor'
# DEVICE_TO_RUN_ON = 'optictensor'
DEVICE_TO_RUN_ON = 'goldentensor'
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 4
NO_OF_ITERATIONS_PER_GPU = 3
NO_OF_ITERATIONS_IN_CPU = 3

dict_run_conditions = {}
# MICROTENSOR CPU RUN
# dict_run_conditions[0] = {'x_dict_size':4,'x_nn_layers':4,'x_nn_nodes':15}
# dict_run_conditions[1] = {'x_dict_size':8,'x_nn_layers':4,'x_nn_nodes':5}
# dict_run_conditions[2] = {'x_dict_size':2,'x_nn_layers':5,'x_nn_nodes':12}
# dict_run_conditions[3] = {'x_dict_size':2,'x_nn_layers':5,'x_nn_nodes':15}
# dict_run_conditions[4] = {'x_dict_size':2,'x_nn_layers':5,'x_nn_nodes':18}
# dict_run_conditions[5] = {'x_dict_size':4,'x_nn_layers':4,'x_nn_nodes':18}


# dict_run_conditions[6] = {'x_dict_size':4,'x_nn_layers':5,'x_nn_nodes':9}
# Runs
# Golden tensor
dict_run_conditions[0] = {'x_dict_size':3,'x_nn_layers':7,'x_nn_nodes':5}
dict_run_conditions[1] = {'x_dict_size':3,'x_nn_layers':7,'x_nn_nodes':10}
dict_run_conditions[2] = {'x_dict_size':3,'x_nn_layers':7,'x_nn_nodes':15}
dict_run_conditions[3] = {'x_dict_size':4,'x_nn_layers':4,'x_nn_nodes':5}

# Optic tensor
# dict_run_conditions[0] = {'x_dict_size':4,'x_nn_layers':3,'x_nn_nodes':5}
# dict_run_conditions[1] = {'x_dict_size':4,'x_nn_layers':3,'x_nn_nodes':10}
# dict_run_conditions[2] = {'x_dict_size':4,'x_nn_layers':3,'x_nn_nodes':15}
# dict_run_conditions[3] = {'x_dict_size':4,'x_nn_layers':4,'x_nn_nodes':10}


# dict_run_conditions[4] = {'x_dict_size':3,'x_nn_layers':4,'x_nn_nodes':18}

oc.write_bash_script(DEVICE_TO_RUN_ON, dict_run_conditions, DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR, NO_OF_ITERATIONS_PER_GPU, NO_OF_ITERATIONS_IN_CPU)

## Transfering the files to the required deepDMD run files
# This is more for organizing all the runs for a single system in a single folder

oc.transfer_current_ocDeepDMD_run_files()

##

SYSTEM_NO = 1
RUN_NO = 25
sess = tf.InteractiveSession()
dict_params,  ls_all_indices, dict_indexed_data,  df_train_learning_curves, df_run_info = oc.get_all_run_info(SYSTEM_NO,RUN_NO,sess)

## Prediction Plots ---  1-step and n-step --- state, output and observables

dict_data_predictions = oc.model_prediction(dict_indexed_data, dict_params, SYSTEM_NO)
# The resulting states and outputs are reverse transformed and hence are unscaled
# For the processing, within the function, x and y are scaled and then the predictions are evaluated

# Plot Parameters - Template 1
individual_plot_width = 4
individual_plot_height = 4
n_state_plots = len(dict_data_predictions[0]['X'][0])
n_outputs = len(dict_data_predictions[0]['Y'][0])
n_graphs_per_row = n_state_plots + n_outputs +1# + len(dict_data_predictions[0]['psiX'][0])# Segregating as states+ouputs and observables
f,ax = plt.subplots(nrows = len(ls_all_indices[0:2]),ncols =n_graphs_per_row , sharex=True, sharey=False, figsize=(individual_plot_width*n_graphs_per_row ,individual_plot_height*len(ls_all_indices[0:2])))

for i in range(len(ls_all_indices[0:2])):
    # States and Output in one graph
    data_index = ls_all_indices[i]
    for j in range(n_state_plots):
        ax[i,j].plot(dict_data_predictions[data_index]['X'][:,j],'*',color=colors[j])
        ax[i,j].plot(dict_data_predictions[data_index]['X_est_one_step'][:, j], '--', color=colors[j])
        ax[i,j].plot(dict_data_predictions[data_index]['X_est_n_step'][:, j], '-', color=colors[j])
    for j in range(n_outputs):
        ax[i,n_state_plots+j].plot(dict_data_predictions[data_index]['Y'][:,j],'*',color=colors[n_state_plots+j])
        ax[i,n_state_plots+j].plot(dict_data_predictions[data_index]['Y_est_one_step'][:, j], '--', color=colors[n_state_plots+j])
        ax[i,n_state_plots+j].plot(dict_data_predictions[data_index]['Y_est_n_step'][:, j], '-', color=colors[n_state_plots+j])
    # Observables in one graph
    for j in range(len(dict_data_predictions[0]['psiX'][0])):
        ax[i,n_state_plots+n_outputs].plot(dict_data_predictions[data_index]['psiX'][:,j],'*',color=colors[j])
        ax[i,n_state_plots+n_outputs].plot(dict_data_predictions[data_index]['psiX_est_one_step'][:, j], '--', color=colors[j])
        ax[i,n_state_plots+n_outputs].plot(dict_data_predictions[data_index]['psiX_est_n_step'][:, j], '-', color=colors[j])
f.show()

# # Plot Parameters - Template 2
# individual_plot_width = 4
# individual_plot_height = 4
# n_graphs_per_row = 2 # Segregating as states+ouputs and observables
# f,ax = plt.subplots(nrows = len(ls_all_indices[0:2]),ncols =n_graphs_per_row , sharex=True, sharey=False, figsize=(individual_plot_width*n_graphs_per_row ,individual_plot_height*len(ls_all_indices[0:2])))
# n_state_plots = len(dict_data_predictions[0]['X'][0])
# for i in range(len(ls_all_indices[0:2])):
#     # States and Output in one graph
#     data_index = ls_all_indices[i]
#     for j in range(n_state_plots):
#         ax[i,0].plot(dict_data_predictions[data_index]['X'][:,j],'*',color=colors[j])
#         ax[i,0].plot(dict_data_predictions[data_index]['X_est_one_step'][:, j], '--', color=colors[j])
#         ax[i,0].plot(dict_data_predictions[data_index]['X_est_n_step'][:, j], '-', color=colors[j])
#     for j in range(len(dict_data_predictions[0]['Y'][0])):
#         ax[i,0].plot(dict_data_predictions[data_index]['Y'][:,j],'*',color=colors[n_state_plots+j])
#         ax[i,0].plot(dict_data_predictions[data_index]['Y_est_one_step'][:, j], '--', color=colors[n_state_plots+j])
#         ax[i,0].plot(dict_data_predictions[data_index]['Y_est_n_step'][:, j], '-', color=colors[n_state_plots+j])
#     # Observables in one graph
#     for j in range(len(dict_data_predictions[0]['psiX'][0])):
#         ax[i,1].plot(dict_data_predictions[data_index]['psiX'][:,j],'*',color=colors[j])
#         ax[i,1].plot(dict_data_predictions[data_index]['psiX_est_one_step'][:, j], '--', color=colors[j])
#         ax[i,1].plot(dict_data_predictions[data_index]['psiX_est_n_step'][:, j], '-', color=colors[j])
# f.show()


## Eigenfunctions and Observables
sampling_resolution = 0.01
dict_psi_phi = oc.observables_and_eigenfunctions(dict_params,sampling_resolution)