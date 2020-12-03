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
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 23
NO_OF_ITERATIONS_PER_GPU = 2
NO_OF_ITERATIONS_IN_CPU = 2
dict_run_conditions = {}

for DEVICE_TO_RUN_ON in ['microtensor','optictensor','goldentensor']:
    if DEVICE_TO_RUN_ON == 'microtensor':
        # MICROTENSOR CPU RUN
        dict_run_conditions[0] = {}
        dict_run_conditions[0]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':3}
        dict_run_conditions[0]['xy'] = {'dict_size':3,'nn_layers':3,'nn_nodes':3}
        dict_run_conditions[1] = {}
        dict_run_conditions[1]['x']  = {'dict_size':9,'nn_layers':7,'nn_nodes':9}
        dict_run_conditions[1]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':6}
        dict_run_conditions[1]['xy'] = {'dict_size':3,'nn_layers':3,'nn_nodes':6}
        dict_run_conditions[2] = {}
        dict_run_conditions[2]['x']  = {'dict_size':9,'nn_layers':9,'nn_nodes':15}
        dict_run_conditions[2]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':9}
        dict_run_conditions[2]['xy'] = {'dict_size':3,'nn_layers':4,'nn_nodes':3}
        dict_run_conditions[3] = {}
        dict_run_conditions[3]['x']  = {'dict_size':9,'nn_layers':9,'nn_nodes':18}
        dict_run_conditions[3]['y']  = {'dict_size':1,'nn_layers':3,'nn_nodes':12}
        dict_run_conditions[3]['xy'] = {'dict_size':3,'nn_layers':4,'nn_nodes':6}
    elif DEVICE_TO_RUN_ON =='goldentensor':
        # Golden tensor
        dict_run_conditions[0] = {}
        dict_run_conditions[0]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':3}
        dict_run_conditions[0]['xy'] = {'dict_size':6,'nn_layers':3,'nn_nodes':6}
        dict_run_conditions[1] = {}
        dict_run_conditions[1]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[1]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':6}
        dict_run_conditions[1]['xy'] = {'dict_size':6,'nn_layers':3,'nn_nodes':9}
        dict_run_conditions[2] = {}
        dict_run_conditions[2]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[2]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':9}
        dict_run_conditions[2]['xy'] = {'dict_size':6,'nn_layers':4,'nn_nodes':6}
        dict_run_conditions[3] = {}
        dict_run_conditions[3]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[3]['y']  = {'dict_size':1,'nn_layers':4,'nn_nodes':12}
        dict_run_conditions[3]['xy'] = {'dict_size':6,'nn_layers':4,'nn_nodes':9}
    elif DEVICE_TO_RUN_ON == 'optictensor':
        # Optic tensor
        dict_run_conditions[0] = {}
        dict_run_conditions[0]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[0]['y']  = {'dict_size':1,'nn_layers':5,'nn_nodes':3}
        dict_run_conditions[0]['xy'] = {'dict_size':9,'nn_layers':3,'nn_nodes':9}
        dict_run_conditions[1] = {}
        dict_run_conditions[1]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[1]['y']  = {'dict_size':1,'nn_layers':5,'nn_nodes':6}
        dict_run_conditions[1]['xy'] = {'dict_size':9,'nn_layers':3,'nn_nodes':12}
        dict_run_conditions[2] = {}
        dict_run_conditions[2]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[2]['y']  = {'dict_size':1,'nn_layers':5,'nn_nodes':9}
        dict_run_conditions[2]['xy'] = {'dict_size':9,'nn_layers':4,'nn_nodes':9}
        dict_run_conditions[3] = {}
        dict_run_conditions[3]['x']  = {'dict_size':9,'nn_layers':6,'nn_nodes':9}
        dict_run_conditions[3]['y']  = {'dict_size':1,'nn_layers':5,'nn_nodes':12}
        dict_run_conditions[3]['xy'] = {'dict_size':9,'nn_layers':4,'nn_nodes':12}
    seq.write_bash_script(DEVICE_TO_RUN_ON, dict_run_conditions, DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR, NO_OF_ITERATIONS_PER_GPU, NO_OF_ITERATIONS_IN_CPU)

## Transfer the oc deepDMD files

seq.transfer_current_ocDeepDMD_run_files()

## RUN 1 PROCESSING - Generate predictions and error
SYSTEM_NO = 23
ls_process_runs = list(range(112,120)) # Runs for which we want to calculate the error
seq.generate_predictions_pickle_file(SYSTEM_NO,state_only =True,ls_process_runs=ls_process_runs)
seq.generate_df_error(SYSTEM_NO,ls_process_runs)

## RUN 1 PROCESSING - Get the optimal run from the specified runs
SYSTEM_NO = 23
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/df_error_SEQUENTIAL.pickle','rb') as handle:
    df_error = pickle.load(handle)
df_error_const_obs = df_error
df_training_plus_validation = df_error_const_obs.train + df_error_const_obs.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))
# opt_run = 37
dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)
print('Optimal Run no: ',opt_run)
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)
# Plotting the fit of the required indices
# SYSTEM_NO = 5
# opt_run = 1
# dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)
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
f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves[0:20],scaled=True,observables=True,one_step=True)
f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_test_curves[0:20],scaled=False,observables=True,one_step=False)

## RUN 1 PROCESSING - Display the hyper parameters
# opt_run = 25 # Use this only if we want to see the hyperparameters of a specific run
with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)
#========================================================================================================================

## OUTPUT STUFF
# Runs error plot
SYSTEM_NO = 23
ls_run_no = list(range(120,144))
plot_params ={}
plot_params['xy_label_font_size']=9
plot_params['individual_fig_width']=2
plot_params['individual_fig_height']=2
seq.plot_training_runs_output(SYSTEM_NO,ls_run_no,plot_params)


## prediction and optimal plots
ls_train_curves = list(range(20))
ls_valid_curves = list(range(20,40))
ls_test_curves = list(range(40,60))
SYSTEM_NO = 23
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
ls_process_run_indices = list(range(144,168))
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

## Info of other output Runs
run_no = 56

dict_predictions_opt_run = seq.get_prediction_data_output(SYSTEM_NO,run_no)
plot_params ={}
plot_params['individual_fig_height'] = 5 #2
plot_params['individual_fig_width'] = 4#2.4
print('Run no: ',run_no)
# f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=False)
f1 = seq.plot_fit_Y(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False)
#
with open(sys_folder_name + '/Sequential/RUN_' + str(run_no) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print(dict_hp)

## ------------------------------------------------------------------------------------------------------------------------------


# Final Runs
SYSTEM_NO = 7
ls_process_runs = list(range(0,72))
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
seq.generate_predictions_pickle_file(SYSTEM_NO,state_only =False,ls_process_runs=ls_process_runs)
seq.generate_df_error_x_and_y(SYSTEM_NO)
with open(sys_folder_name + '/df_error_SEQUENTIAL_x_and_y.pickle','rb') as handle:
    df_error = pickle.load(handle)
df_training_plus_validation = df_error.train + df_error.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))

with open(sys_folder_name + '/Sequential/RUN_' + str(opt_run) + '/dict_hyperparameters.pickle','rb') as handle:
    dict_hp = pickle.load(handle)
print('Optimal Run Hyperparameters')
print(dict_hp)

dict_predictions_opt_run = seq.get_prediction_data(SYSTEM_NO,opt_run)
plot_params ={}
plot_params['individual_fig_height'] = 5 #2
plot_params['individual_fig_width'] = 4#2.4
ls_train_curves = list(range(20))
ls_valid_curves = list(range(20,40))
ls_test_curves = list(range(40,60))
# f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=True,observables=True)
f1 = seq.plot_fit_XY(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False,observables=True)
## Plotting the observables
plot_params={}
plot_params['xy_label_font_size']=15
plot_params['individual_fig_width']=15
plot_params['individual_fig_height']=15
f2 = seq.plot_observables(dict_predictions_opt_run,plot_params)

## Saving the Optimal first run result
SYSTEM_NO = 23
RUN_NO = 115
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    d = pickle.load(handle)
with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
    d1 = pickle.load(handle)
for items in d1.keys():
    d[items] = d1[items]
print(d.keys())
with open('/Users/shara/Desktop/oc_deepDMD/System_23_BestRun_1.pickle','wb') as handle:
    pickle.dump(d,handle)

## Saving the Optimal Results of the Second Run
SYSTEM_NO = 23
RUN_NO = 153
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    d = pickle.load(handle)
with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
    d1 = pickle.load(handle)
for items in d1.keys():
    d[items] = d1[items]
print(d.keys())
with open('/Users/shara/Desktop/oc_deepDMD/System_23_BestRun_2.pickle','wb') as handle:
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

##
ls_filtered_runs = ls_process_runs
## RUN 1/3 - upto N-step prediction error calculation for all the specified runs (ls_filtered_runs)
SYSTEM_NO = 23
ls_steps = list(range(10,200,10))
# Get the data

sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'rb') as handle:
    dict_indexed_data = pickle.load(handle)
dict_rmse ={}
dict_r2={}
for run_i in ls_filtered_runs:
    print('RUN: ',run_i)
    run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(run_i)
    sess = tf.InteractiveSession()
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    psixfT = tf.get_collection('psixfT')[0]
    xfT_feed = tf.get_collection('xfT_feed')[0]
    KxT = tf.get_collection('KxT')[0]
    KxT_num = sess.run(KxT)
    dict_rmse_run = {}
    dict_r2_run = {}
    for data_index in dict_indexed_data.keys():  # iterating through each dataset
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_indexed_data[data_index], SYSTEM_NO)
        X_scaled = dict_DATA_i['X']
        psiX = psixfT.eval(feed_dict={xfT_feed: X_scaled})
        dict_rmse_run[data_index] = {}
        dict_r2_run[data_index] = {}
        for i in ls_steps:  # iterating through each step prediction
            np_psiX_true = psiX[i:, :]
            np_psiX_pred = np.matmul(psiX[:-i, :],np.linalg.matrix_power(KxT_num, i))  # i step prediction at each datapoint
            dict_rmse_run[data_index][i] = np.sqrt(np.mean(np.square(np_psiX_true - np_psiX_pred)))
            dict_r2_run[data_index][i] = np.max([0, (1 - np.sum(np.square(np_psiX_true - np_psiX_pred)) / np.sum(np.square(np_psiX_true))) * 100])
    dict_rmse[run_i]= pd.DataFrame(dict_rmse_run).mean(axis=1).to_dict()
    dict_r2[run_i]= pd.DataFrame(dict_r2_run).mean(axis=1).to_dict()
    tf.reset_default_graph()
    sess.close()
print(pd.DataFrame(dict_r2))


## SYSTEM 1 ANALYSIS
SYSTEM_NO = 7
ls_process_runs = list(range(0, 72))
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
os.listdir(sys_folder_name + '/Sequential')

# # Get the unique observables
# for run in ls_process_runs:
#     with open(sys_folder_name + 'Sequential/RUN_' + str(run) + '/dict_hyperparameters.pickle','rb') as handle:
#         d =