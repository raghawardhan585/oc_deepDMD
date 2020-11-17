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

def generate_hyperparameter_dataframe(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    dict_hyperparameters = {}
    RUNS = []
    for items in os.listdir(sys_folder_name):
        if items[0:4] == 'RUN_':
            RUN_NO = int(items[4:])
            RUNS.append(RUN_NO)
            run_folder_name = sys_folder_name + '/RUN_' + str(RUN_NO)
            with open(run_folder_name + '/run_info.pickle', 'rb') as handle:
                df_run_info = pd.DataFrame(pickle.load(handle))
            n_nodes = df_run_info.loc['x_hidden_variable_list', df_run_info.columns[-1]][0]
            n_layers = len(df_run_info.loc['x_hidden_variable_list', df_run_info.columns[-1]])
            n_observables = df_run_info.loc['x_hidden_variable_list', df_run_info.columns[-1]][-1]
            # training_error = df_run_info.loc['training error', df_run_info.columns[-1]]
            # validation_error = df_run_info.loc['validation error', df_run_info.columns[-1]]
            dict_hyperparameters [RUN_NO] = {'n_nodes': n_nodes, 'n_layers': n_layers, 'n_observables': n_observables}
                                       # ,'training_error': training_error, 'validation_error': validation_error}
    df_hyperparameters = pd.DataFrame(dict_hyperparameters).T.sort_index()
    with open(sys_folder_name + '/df_hyperparameters.pickle','wb') as handle:
        pickle.dump(df_hyperparameters,handle)
    return


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
        try:
            sampling_resolution = 0.01
            dict_psi_phi = oc.observables_and_eigenfunctions(dict_params, sampling_resolution)
            dict_predictions[run]['X1'] = dict_psi_phi['X1']
            dict_predictions[run]['X2'] = dict_psi_phi['X2']
            dict_predictions[run]['observables'] = dict_psi_phi['observables']
            dict_predictions[run]['eigenfunctions'] = dict_psi_phi['eigenfunctions']
        except:
            print('Cannot find the eigenfunctions and observables as the number of base states is not equal to 2')
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
        all_errors = np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step'])
        # all_errors = np.append(np.square(dict_XY[i]['X'] - dict_XY[i]['X_est_n_step']) , np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step']))
        # all_errors = np.append(all_errors, np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step']))
        J_error = np.append(J_error, np.mean(all_errors))
    # J_error = np.log10(np.max(J_error))
    J_error = np.mean(J_error)
    return J_error

def generate_df_error(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
        SYSTEM_NO)
    ls_train_curves = list(range(20))
    ls_valid_curves = list(range(20, 40))
    ls_test_curves = list(range(40, 60))
    with open(sys_folder_name + '/dict_predictions.pickle', 'rb') as handle:
        dict_predictions = pickle.load(handle)
    dict_error = {}
    for run_no in dict_predictions.keys():
        print(run_no)
        dict_error[run_no] = {}
        dict_error[run_no]['train'] = get_error(ls_train_curves,dict_predictions[run_no])
        dict_error[run_no]['valid'] = get_error(ls_valid_curves, dict_predictions[run_no])
        dict_error[run_no]['test'] = get_error(ls_test_curves, dict_predictions[run_no])
    df_error = pd.DataFrame(dict_error).T
    # Save the file
    if os.path.exists(sys_folder_name + '/df_error.pickle'):
        ip = input('Do you wanna write over the df_error file[y/n]?')
        if ip == 'y':
            os.remove(sys_folder_name + '/df_error.pickle')
            with open(sys_folder_name + '/df_error.pickle', 'wb') as handle:
                pickle.dump(df_error, handle)
    else:
        with open(sys_folder_name + '/df_error.pickle','wb') as handle:
            pickle.dump(df_error,handle)
    return

def plot_fit_XY(dict_run,plot_params,ls_runs,scaled=False,observables=False):
    n_rows = 7
    if observables:
         n_cols = 9
         graphs_per_run = 3
    else:
        n_cols = 6
        graphs_per_run = 2
    f,ax = plt.subplots(n_rows,n_cols,sharex=True,figsize = (plot_params['individual_fig_width']*n_cols,plot_params['individual_fig_height']*n_rows))
    i = 0
    for row_i in range(n_rows):
        for col_i in list(range(0,n_cols,graphs_per_run)):
            if scaled:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X_scaled'].shape[1]
                for j in range(n_states):
                    ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled'][:, j], '.', color=colors[j])
                    ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled_est_n_step'][:, j], color=colors[j],
                                          label='x' + str(j + 1)+ '[scaled]')
                ax[row_i, col_i].legend()
                for j in range(dict_run[ls_runs[i]]['Y_scaled'].shape[1]):
                    ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled'][:, j], '.', color=colors[n_states + j])
                    ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled_est_n_step'][:, j], color=colors[n_states + j],
                                              label='y' + str(j + 1)+ '[scaled]')
                ax[row_i, col_i + 1].legend()
            else:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X'].shape[1]
                for j in range(n_states):
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X'][:,j],'.',color = colors[j])
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X_est_n_step'][:, j], color=colors[j],label ='x'+str(j+1) )
                ax[row_i, col_i].legend()
                for j in range(dict_run[ls_runs[i]]['Y'].shape[1]):
                    ax[row_i,col_i+1].plot(dict_run[ls_runs[i]]['Y'][:,j],'.',color = colors[n_states+j])
                    ax[row_i,col_i+1].plot(dict_run[ls_runs[i]]['Y_est_n_step'][:, j], color=colors[n_states+j],label ='y'+str(j+1))
                ax[row_i, col_i+1].legend()
            if observables:
                # Plot the observables
                for j in range(dict_run[ls_runs[i]]['psiX'].shape[1]):
                    ax[row_i,col_i+2].plot(dict_run[ls_runs[i]]['psiX'][:,j],'.',color = colors[np.mod(j,7)],linewidth = int(j/7+1))
                    ax[row_i,col_i+2].plot(dict_run[ls_runs[i]]['psiX_est_n_step'][:, j], color=colors[np.mod(j,7)],linewidth = int(j/7+1),label ='x_'+str(j+1) )
                ax[row_i, col_i+2].legend()
            i = i+1
            if i == len(ls_runs):
                break
    f.show()
    return f


def plot_observables(dict_run,plot_params):
    # x horizontal y vertical
    # n_x = int(np.ceil(np.sqrt(dict_run['observables'].shape[2])))
    # n_y = int(np.ceil(dict_run['observables'].shape[2]/n_x))
    n_x = dict_run['observables'].shape[2]
    n_y = 1
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

def get_prediction_data(SYSTEM_NO,RUN_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    with open(sys_folder_name + '/dict_predictions.pickle', 'rb') as handle:
        dict_predictions = pickle.load(handle)
    return dict_predictions[RUN_NO]
## MAIN
SYSTEM_NO = 5
ls_train_curves = list(range(20))
ls_valid_curves = list(range(20,40))
ls_test_curves = list(range(40,60))
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)

generate_predictions_pickle_file(SYSTEM_NO)
generate_df_error(SYSTEM_NO)
generate_hyperparameter_dataframe(SYSTEM_NO)
# with open(sys_folder_name + '/dict_predictions.pickle', 'rb') as handle:
#     dict_predictions = pickle.load(handle)
# with open(sys_folder_name + '/df_error.pickle','rb') as handle:
#     df_error = pickle.load(handle)


# df_e = df_error.loc[df_error.train<5]
# plt.figure()
# plt.plot(df_e.index,df_e.iloc[:,0:2].sum(axis=1))
# plt.legend(['Training Error', 'Validation Error'])
# plt.xlabel('Run Number')
# plt.ylabel('log(max(error))')
# plt.show()
## Get the optimal run for the given number of observables
SYSTEM_NO = 5
N_OBSERVABLES = 3
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/df_hyperparameters.pickle', 'rb') as handle:
    df_hyperparameters = pickle.load(handle)
df_hyp_const_obs = df_hyperparameters[df_hyperparameters.n_observables==N_OBSERVABLES]
ls_runs_const_obs = list(df_hyp_const_obs.index)
with open(sys_folder_name + '/df_error.pickle','rb') as handle:
    df_error = pickle.load(handle)
ls_runs_const_obs = list(range(37,40))

# Check is
ls_all_runs = list(df_hyperparameters.index)
for items in ls_runs_const_obs:
    if items not in ls_all_runs:
        ls_runs_const_obs.remove(items)


df_error_const_obs = df_error.loc[ls_runs_const_obs,:]
# df_error_const_obs = df_error
df_training_plus_validation = df_error_const_obs.train + df_error_const_obs.valid
opt_run = int(np.array(df_training_plus_validation.loc[df_training_plus_validation == df_training_plus_validation .min()].index))
opt_run = 37
dict_predictions_opt_run = get_prediction_data(SYSTEM_NO,opt_run)






## Plotting the fit of the required indices
plot_params ={}
plot_params['individual_fig_height'] = 2
plot_params['individual_fig_width'] = 2.4
# f1 = plot_fit_XY(dict_predictions_opt_run,plot_params,ls_train_curves,scaled=True,observables=False)
f1 = plot_fit_XY(dict_predictions_opt_run,plot_params,ls_test_curves,scaled=False,observables=False)
##
dict_run = dict_predictions_opt_run
ls_runs = ls_test_curves
scaled = True
observables = False

##

n_rows = 7
if observables:
     n_cols = 9
     graphs_per_run = 3
else:
    n_cols = 6
    graphs_per_run = 2
f,ax = plt.subplots(n_rows,n_cols,sharex=True,figsize = (plot_params['individual_fig_width']*n_cols,plot_params['individual_fig_height']*n_rows))
i = 0
for row_i in range(n_rows):
    for col_i in list(range(0,n_cols,graphs_per_run)):
        if scaled:
            # Plot states and outputs
            n_states = dict_run[ls_runs[i]]['X_scaled'].shape[1]
            for j in range(n_states):
                ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled'][:, j], '.', color=colors[j])
                ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled_est_n_step'][:, j], color=colors[j],
                                      label='x' + str(j + 1)+ '[scaled]')
            ax[row_i, col_i].legend()
            for j in range(dict_run[ls_runs[i]]['Y_scaled'].shape[1]):
                ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled'][:, j], '.', color=colors[n_states + j])
                ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled_est_n_step'][:, j], color=colors[n_states + j],
                                          label='y' + str(j + 1)+ '[scaled]')
            ax[row_i, col_i + 1].legend()
        else:
            # Plot states and outputs
            n_states = dict_run[ls_runs[i]]['X'].shape[1]
            for j in range(n_states):
                ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X'][:,j],'.',color = colors[j])
                ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X_est_n_step'][:, j], color=colors[j],label ='x'+str(j+1) )
            ax[row_i, col_i].legend()
            for j in range(dict_run[ls_runs[i]]['Y'].shape[1]):
                ax[row_i,col_i+1].plot(dict_run[ls_runs[i]]['Y'][:,j],'.',color = colors[n_states+j])
                ax[row_i,col_i+1].plot(dict_run[ls_runs[i]]['Y_est_n_step'][:, j], color=colors[n_states+j],label ='y'+str(j+1))
            ax[row_i, col_i+1].legend()
        if observables:
            # Plot the observables
            for j in range(dict_run[ls_runs[i]]['psiX'].shape[1]):
                ax[row_i,col_i+2].plot(dict_run[ls_runs[i]]['psiX'][:,j],'.',color = colors[np.mod(j,7)],linewidth = int(j/7+1))
                ax[row_i,col_i+2].plot(dict_run[ls_runs[i]]['psiX_est_n_step'][:, j], color=colors[np.mod(j,7)],linewidth = int(j/7+1),label ='x_'+str(j+1) )
            ax[row_i, col_i+2].legend()
        i = i+1
        if i == len(ls_runs):
            break
f.show()

## Plotting the observables
plot_params={}
plot_params['xy_label_font_size']=15
plot_params['individual_fig_width']=15
plot_params['individual_fig_height']=15
f2 = plot_observables(dict_predictions_opt_run,plot_params)


