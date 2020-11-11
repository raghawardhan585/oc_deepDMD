##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import ocdeepdmd_simulation_examples_helper_functions as oc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import shutil

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette

def generate_predictions_pickle_file(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    # Make a predictions folder if one doesn't exist
    if os.path.exists(sys_folder_name + '/dict_predictions.pickle'):
        with open(sys_folder_name + '/dict_predictions.pickle','rb') as handle:
            dict_predictions = pickle.load(handle)
    else:
        dict_predictions = {}
    # Figure out the unprocessed food
    ls_processed_runs = list(dict_predictions.keys())
    # Scan all folders to get all Run Indices
    ls_all_run_indices =[]
    for folder in os.listdir(sys_folder_name):
        if folder[0:4] == 'RUN_': # It is a RUN folder
            ls_all_run_indices.append(int(folder[4:]))
    ls_unprocessed_runs = list(set(ls_all_run_indices) - set(ls_processed_runs))
    print('RUNS TO PROCESS - ',ls_unprocessed_runs)
    # Updating the dictionary of predictions
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
    # Saving the dict_predictions folder
    with open(sys_folder_name + '/dict_predictions.pickle','wb') as handle:
        pickle.dump(dict_predictions,handle)
    return

def get_error(ls_indices,dict_XY):
    J_error = np.empty(shape=(0,1))
    for i in ls_indices:
        all_errors = np.append(np.square(dict_XY[i]['X'] - dict_XY[i]['X_est_n_step']) , np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step']))
        all_errors = np.append(all_errors, np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step']))
        J_error = np.append(J_error, np.mean(all_errors))
    J_error = np.log10(np.max(J_error))
    # J_error = np.mean(J_error)
    return J_error

def generate_dict_error(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
        SYSTEM_NO)
    ls_train_runs = list(range(20))
    ls_valid_runs = list(range(20, 40))
    ls_test_runs = list(range(40, 60))
    with open(sys_folder_name + '/dict_predictions.pickle', 'rb') as handle:
        dict_predictions = pickle.load(handle)
    dict_error = {}
    for run_no in dict_predictions.keys():
        print(run_no)
        dict_error[run_no] = {}
        dict_error[run_no]['train'] = get_error(ls_train_runs,dict_predictions[run_no])
        dict_error[run_no]['valid'] = get_error(ls_valid_runs, dict_predictions[run_no])
        dict_error[run_no]['test'] = get_error(ls_test_runs, dict_predictions[run_no])
    # Save the file
    if os.path.exists(sys_folder_name + '/dict_error.pickle'):
        ip = input('Do you wanna write over the dict_error file[y/n]?')
        if ip == 'y':
            shutil.rmtree(sys_folder_name + '/dict_error.pickle')
            with open(sys_folder_name + '/dict_error.pickle', 'wb') as handle:
                pickle.dump(dict_error, handle)
    else:
        with open(sys_folder_name + '/dict_error.pickle','wb') as handle:
            pickle.dump(dict_error,handle)
    return

def plot_observables(dict_run):
    # x horizontal y vertical
    n_x = int(np.ceil(np.sqrt(dict_run['observables'].shape[2])))
    n_y = int(np.ceil(dict_run['observables'].shape[2]/n_x))
    fig = plt.figure(figsize=(plot_params['individual_fig_width']*n_x,plot_params['individual_fig_height']*n_y))
    for i in range(dict_run['observables'].shape[2]):
        ax = fig.add_subplot(n_y,n_x, i + 1, projection='3d')
        ax.plot_surface(dict_run['X1'], dict_run['X2'], dict_run['observables'][:,:,i], cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # if np.sum(dict_optrun['observables'][:, :, i]) == 0:
        #     ax.set_title('$\psi_{}$'.format(i + 1) + '(x) = 0')
        # else:
        #     ax.set_title('$\psi_{}$'.format(i + 1) + '(x)')
        # ax.title.set_fontsize(9)
        ax.grid(False)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.xaxis.label.set_fontsize(plot_params['xy_label_font_size'])
        ax.yaxis.label.set_fontsize(plot_params['xy_label_font_size'])
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
    fig.show()
    return fig
## MAIN
SYSTEM_NO = 2
ls_train_runs = list(range(20))
ls_valid_runs = list(range(20,40))
ls_test_runs = list(range(40,60))
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)

# generate_predictions_pickle_file(SYSTEM_NO)
# generate_dict_error(SYSTEM_NO)
# with open(sys_folder_name + '/dict_predictions.pickle', 'rb') as handle:
#     dict_predictions = pickle.load(handle)
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
dict_optrun = dict_predictions[opt_run]
if os.path.exists(sys_folder_name + '/dict_optrun.pickle'):
    ip = input('Do you wanna write over the dict_optrun file[y/n]?')
    if ip == 'y':
        shutil.rmtree(sys_folder_name + '/dict_optrun.pickle')
        with open(sys_folder_name + '/dict_optrun.pickle', 'wb') as handle:
            pickle.dump(dict_optrun, handle)
else:
    with open(sys_folder_name + '/dict_optrun.pickle','wb') as handle:
        pickle.dump(dict_optrun,handle)

## Optimal Run Results
SYSTEM_NO = 2
ls_test_runs = list(range(40,60))
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/dict_optrun.pickle','rb') as handle:
    dict_optrun = pickle.load(handle)
fig_height = 1
fig_width = 2
f,ax = plt.subplots(7,6,sharex=True,figsize = (fig_width*6,fig_height*7))
i = 0
# dict_optrun = dict_predictions[opt_run]
for row_i in range(7):
    for col_i in list(range(0,6,2)):
        # Plot states and outputs
        n_states = dict_optrun[ls_test_runs[i]]['X'].shape[1]
        for j in range(n_states):
            ax[row_i,col_i].plot(dict_optrun[ls_test_runs[i]]['X'][:,j],'.',color = colors[j])
            ax[row_i,col_i].plot(dict_optrun[ls_test_runs[i]]['X_est_n_step'][:, j], color=colors[j],label ='x_'+str(j+1) )
        ax[row_i, col_i].legend()
        for j in range(dict_optrun[ls_test_runs[i]]['Y'].shape[1]):
            ax[row_i,col_i+1].plot(dict_optrun[ls_test_runs[i]]['Y'][:,j],'.',color = colors[n_states+j])
            ax[row_i,col_i+1].plot(dict_optrun[ls_test_runs[i]]['Y_est_n_step'][:, j], color=colors[n_states+j],label ='y_'+str(j+1))
        ax[row_i, col_i+1].legend()
        # ax[row_i, col_i].title('X,Y')
        # Plot the observables
        # for j in range(dict_optrun[ls_test_runs[i]]['psiX'].shape[1]):
        #     ax[row_i,col_i+1].plot(dict_optrun[ls_test_runs[i]]['psiX'][:,j],'.',color = colors[np.mod(j,7)])
        #     ax[row_i,col_i+1].plot(dict_optrun[ls_test_runs[i]]['psiX_est_n_step'][:, j], color=colors[np.mod(j,7)],label ='x_'+str(j+1) )
        # ax[row_i, col_i].title('observables')
        i = i+1
        if i == len(ls_test_runs):
            break

f.show()

## Plotting the observables
plot_params={}
plot_params['xy_label_font_size']=15
plot_params['individual_fig_width']=15
plot_params['individual_fig_height']=15
f = plot_observables(dict_optrun,plot_params)


