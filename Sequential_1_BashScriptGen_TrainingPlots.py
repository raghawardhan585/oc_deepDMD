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
# DEVICE_TO_RUN_ON = 'microtensor'
# DEVICE_TO_RUN_ON = 'optictensor'
DEVICE_TO_RUN_ON = 'goldentensor'
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 6
NO_OF_ITERATIONS_PER_GPU = 2
NO_OF_ITERATIONS_IN_CPU = 2
dict_run_conditions = {}

# MICROTENSOR CPU RUN
# dict_run_conditions[0] = {}
# dict_run_conditions[0]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':3}
# dict_run_conditions[0]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[1] = {}
# dict_run_conditions[1]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[1]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':6}
# dict_run_conditions[1]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[2] = {}
# dict_run_conditions[2]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[2]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':9}
# dict_run_conditions[2]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[3] = {}
# dict_run_conditions[3]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[3]['y']  = {'dict_size':3,'nn_layers':4,'nn_nodes':3}
# dict_run_conditions[3]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}

# Golden tensor
dict_run_conditions[0] = {}
dict_run_conditions[0]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
dict_run_conditions[0]['y']  = {'dict_size':6,'nn_layers':4,'nn_nodes':9}
dict_run_conditions[0]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[1] = {}
dict_run_conditions[1]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
dict_run_conditions[1]['y']  = {'dict_size':9,'nn_layers':4,'nn_nodes':3}
dict_run_conditions[1]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[2] = {}
dict_run_conditions[2]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
dict_run_conditions[2]['y']  = {'dict_size':9,'nn_layers':4,'nn_nodes':6}
dict_run_conditions[2]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
dict_run_conditions[3] = {}
dict_run_conditions[3]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
dict_run_conditions[3]['y']  = {'dict_size':9,'nn_layers':4,'nn_nodes':9}
dict_run_conditions[3]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}

# Optic tensor
# dict_run_conditions[0] = {}
# dict_run_conditions[0]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[0]['y']  = {'dict_size':3,'nn_layers':4,'nn_nodes':6}
# dict_run_conditions[0]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[1] = {}
# dict_run_conditions[1]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[1]['y']  = {'dict_size':3,'nn_layers':4,'nn_nodes':9}
# dict_run_conditions[1]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[2] = {}
# dict_run_conditions[2]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[2]['y']  = {'dict_size':6,'nn_layers':4,'nn_nodes':3}
# dict_run_conditions[2]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}
# dict_run_conditions[3] = {}
# dict_run_conditions[3]['x']  = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
# dict_run_conditions[3]['y']  = {'dict_size':6,'nn_layers':4,'nn_nodes':6}
# dict_run_conditions[3]['xy'] = {'dict_size':2,'nn_layers':3,'nn_nodes':3}

seq.write_bash_script(DEVICE_TO_RUN_ON, dict_run_conditions, DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR, NO_OF_ITERATIONS_PER_GPU, NO_OF_ITERATIONS_IN_CPU)

## TRansfer the oc deepDMD files

seq.transfer_current_ocDeepDMD_run_files()

##
SYSTEM_NO = 6
seq.generate_predictions_pickle_file(SYSTEM_NO,state_only =True,ls_process_runs=list(range(24,42)))
seq.generate_df_error(SYSTEM_NO)
# seq.generate_hyperparameter_dataframe(SYSTEM_NO)


## Get the optimal run for the given number of observables

# dict_filter_criteria={}
# dict_filter_criteria['x'] = {'obs':[],}



SYSTEM_NO = 6
# N_OBSERVABLES = 3
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# with open(sys_folder_name + '/df_hyperparameters.pickle', 'rb') as handle:
#     df_hyperparameters = pickle.load(handle)
# df_hyp_const_obs = df_hyperparameters[df_hyperparameters.n_observables==N_OBSERVABLES]
# ls_runs_const_obs = list(df_hyp_const_obs.index)
with open(sys_folder_name + '/df_error_SEQUENTIAL.pickle','rb') as handle:
    df_error = pickle.load(handle)
# ls_runs_const_obs = list(range(19))

# Check is
# ls_all_runs = list(df_hyperparameters.index)
# for items in ls_runs_const_obs:
#     if items not in ls_all_runs:
#         ls_runs_const_obs.remove(items)


# df_error_const_obs = df_error.loc[ls_runs_const_obs,:]
df_error_const_obs = df_error
df_training_plus_validation = df_error_const_obs.train + df_error_const_obs.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))
# opt_run = 37
dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)
print('Optimal Run no: ',opt_run)
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)
## Plotting the fit of the required indices
# SYSTEM_NO = 5
# opt_run = 1
# dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)
plot_params ={}
plot_params['individual_fig_height'] = 5 #2
plot_params['individual_fig_width'] = 4#2.4
ls_train_curves = list(range(20))
ls_valid_curves = list(range(20,40))
ls_test_curves = list(range(40,60))
# f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=True,observables=True)
f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False,observables=True)

##
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)


## OUTPUT STUFF
# Runs error plot
SYSTEM_NO = 6
ls_run_no = list(range(0,24)) #18
plot_params ={}
plot_params['xy_label_font_size']=9
plot_params['individual_fig_width']=2
plot_params['individual_fig_height']=2
seq.plot_training_runs_output(SYSTEM_NO,ls_run_no,plot_params)


## prediction and optimal plots
ls_train_curves = list(range(20))
ls_valid_curves = list(range(20,40))
ls_test_curves = list(range(40,60))
SYSTEM_NO = 6
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
ls_process_run_indices = list(range(0,24))
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
##
# f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=False)
f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False)
#
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)