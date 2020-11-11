##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import ocdeepdmd_simulation_examples_helper_functions as oc
import shutil

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


SYSTEM_NO = 2
ls_train_runs = list(range(20))
ls_valid_runs = list(range(20,40))
ls_test_runs = list(range(40,60))
N_VALID_RUNS = 20

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
##
def get_error(ls_indices,dict_XY):
    J_error = np.empty(shape=(0,1))
    for i in ls_indices:
        all_errors = np.append(np.square(dict_XY[i]['X'] - dict_XY[i]['X_est_n_step']) , np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step']))
        all_errors = np.append(all_errors, np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step']))
        J_error = np.append(J_error, np.mean(all_errors))
    J_error = np.log10(np.max(J_error))
    # J_error = np.mean(J_error)
    return J_error

dict_error = {}
for run_no in dict_predictions.keys():
    print(run_no)
    dict_error[run_no] ={}
    dict_error[run_no]['train'] = get_error(ls_train_runs,dict_predictions[run_no])
    dict_error[run_no]['valid'] = get_error(ls_valid_runs, dict_predictions[run_no])
    dict_error[run_no]['test'] = get_error(ls_test_runs, dict_predictions[run_no])


if os.path.exists(sys_folder_name + '/dict_error.pickle'):
    ip = input('Do you wanna write over the dict_error file[y/n]?')
    if ip == 'y':
        shutil.rmtree(sys_folder_name + '/dict_error.pickle')
        with open(sys_folder_name + '/dict_error.pickle', 'wb') as handle:
            pickle.dump(dict_error, handle)
else:
    with open(sys_folder_name + '/dict_error.pickle','wb') as handle:
        pickle.dump(dict_error,handle)
##
SYSTEM_NO = 2
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/dict_error.pickle','rb') as handle:
    dict_error = pickle.load(handle)
df_error = pd.DataFrame(dict_error).T
df_e = df_error.loc[df_error.train<5]
plt.figure()
plt.plot(df_e.index,df_e.iloc[:,0:2].sum(axis=1))
plt.legend(['Training Error', 'Validation Error'])
plt.xlabel('Run Number')
plt.ylabel('log(max(error))')
plt.show()
##
df_training_plus_validation = df_error.train + df_error.valid

opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))




