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
import ocdeepdmd_simulation_examples_helper_functions as oc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


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
    destination_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NUMBER) + '/deepDMD'
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

def get_all_run_info(SYSTEM_NO,RUN_NO,sess):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    run_folder_name = sys_folder_name + '/deepDMD/RUN_' + str(RUN_NO)
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    with open(run_folder_name + '/all_tf_var_names.pickle', 'rb') as handle:
        d = pickle.load(handle)
    dict_params = {}
    for items in d:
        dict_params[items] = tf.get_collection(items)[0]
    try:
        dict_params['KxT_num'] = sess.run(dict_params['KxT'])
    except:
        print('Error in State Transition Matrix')
    try:
        dict_params['WhT_num'] = sess.run(dict_params['WhT'])
    except:
        print('Error in Output Matrix')
    return dict_params

def generate_predictions_pickle_file(SYSTEM_NO, ls_process_runs):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    # -----------------------------------Get the required data
    with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'rb') as handle:
        dict_indexed_data = pickle.load(handle)  # No scaling appplied here
    # -----------------------------------Make a predictions folder if one doesn't exist
    if os.path.exists(sys_folder_name + '/dict_predictions_deepDMD.pickle'):
        with open(sys_folder_name + '/dict_predictions_deepDMD.pickle','rb') as handle:
            dict_predictions_deepDMD = pickle.load(handle)
    else:
        dict_predictions_deepDMD = {}
    # -----------------------------------Find all runs to be processed
    ls_all_run_indices = []
    for folder in os.listdir(sys_folder_name+'/deepDMD'):
        if folder[0:4] == 'RUN_': # It is a RUN folder
            ls_all_run_indices.append(int(folder[4:]))
    ls_processed_runs = list(dict_predictions_deepDMD.keys())
    ls_unprocessed_runs = list(set(ls_all_run_indices) - set(ls_processed_runs))
    if len(ls_process_runs) !=0:
        ls_unprocessed_runs = list(set(ls_unprocessed_runs).intersection(set(ls_process_runs)))
    # ----------------------------------- Process the runs possible
    print('RUNS TO PROCESS - ',ls_unprocessed_runs)
    for run in ls_unprocessed_runs:
        print('Run: ',run)
        run_folder_name = sys_folder_name + '/deepDMD/RUN_' + str(run)
        dict_predictions_deepDMD[run] = {}
        sess = tf.InteractiveSession()
        dict_params = get_all_run_info(SYSTEM_NO, run, sess)
        # Get the 1-step and n-step prediction data
        for data_index in dict_indexed_data.keys():
            dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_indexed_data[data_index], SYSTEM_NO)
            X_scaled = dict_DATA_i['X']
            Y_scaled = dict_DATA_i['Y']
            psiX =  dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled})
            # 1 - step predictions
            psiX_1step = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[0:1,:]})
            psiX_1step = np.concatenate([psiX_1step,np.matmul(dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[0:-1, :]}),dict_params['KxT_num'])],axis=0)
            Y_1step_scaled = np.matmul(psiX_1step,dict_params['WhT_num'])
            # N - step predictions
            psiX_nstep = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[0:1,:]})
            for i in range(1,X_scaled.shape[0]):
                psiX_nstep = np.concatenate([psiX_nstep, np.matmul(psiX_nstep[-1:],dict_params['KxT_num'])], axis=0)
            Y_nstep_scaled = np.matmul(psiX_nstep, dict_params['WhT_num'])
            dict_predictions_deepDMD[run][data_index] = {'X': dict_indexed_data[data_index]['X'], 'Y': dict_indexed_data[data_index]['Y'], 'psiX': psiX, 'X_scaled': X_scaled, 'Y_scaled': Y_scaled}
            dict_predictions_deepDMD[run][data_index]['psiX_one_step_scaled'] = psiX_1step
            dict_predictions_deepDMD[run][data_index]['psiX_n_step_scaled'] = psiX_nstep
            dict_predictions_deepDMD[run][data_index]['X_one_step_scaled'] = psiX_1step[:,0:len(X_scaled[0])]
            dict_predictions_deepDMD[run][data_index]['X_n_step_scaled'] = psiX_nstep[:,0:len(X_scaled[0])]
            dict_predictions_deepDMD[run][data_index]['X_one_step'] = oc.inverse_transform_X(psiX_1step[:, 0:len(X_scaled[0])], SYSTEM_NO)
            dict_predictions_deepDMD[run][data_index]['X_n_step'] = oc.inverse_transform_X(psiX_nstep[:, 0:len(X_scaled[0])], SYSTEM_NO)
            dict_predictions_deepDMD[run][data_index]['Y_one_step_scaled'] = Y_1step_scaled
            dict_predictions_deepDMD[run][data_index]['Y_n_step_scaled'] = Y_nstep_scaled
            dict_predictions_deepDMD[run][data_index]['Y_one_step'] = oc.inverse_transform_Y(Y_1step_scaled, SYSTEM_NO)
            dict_predictions_deepDMD[run][data_index]['Y_n_step'] = oc.inverse_transform_Y(Y_nstep_scaled, SYSTEM_NO)
        tf.reset_default_graph()
        sess.close()
    # Saving the dict_predictions folder
    with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'wb') as handle:
        pickle.dump(dict_predictions_deepDMD, handle)
    return

def get_error(ls_indices,dict_XY):
    J_error = np.empty(shape=(0,1))
    for i in ls_indices:
        # all_errors = np.square(dict_XY[i]['X_scaled'] - dict_XY[i]['X_one_step_scaled'])
        # all_errors = np.concatenate([dict_XY[i]['X'] - dict_XY[i]['X_one_step'],dict_XY[i]['Y'] - dict_XY[i]['Y_one_step']],axis=1)
        # all_errors = np.concatenate([dict_XY[i]['X_scaled'] - dict_XY[i]['X_one_step_scaled'],dict_XY[i]['Y_scaled'] - dict_XY[i]['Y_one_step_scaled']],axis=1)
        all_errors = np.concatenate([dict_XY[i]['X_scaled'] - dict_XY[i]['X_n_step_scaled'],dict_XY[i]['Y_scaled'] - dict_XY[i]['Y_n_step_scaled']], axis=1)
        # all_errors = np.concatenate([dict_XY[i]['X'] - dict_XY[i]['X_n_step'], dict_XY[i]['Y'] - dict_XY[i]['Y_n_step']], axis=1)
        # all_errors = np.append(all_errors, np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step']))
        J_error = np.append(J_error, np.mean(np.square(all_errors)))
    # J_error = np.log10(np.max(J_error))
    J_error = np.mean(J_error)
    return J_error


def get_run_performance_stats(SYSTEM_NO,RUN_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    with open(sys_folder_name + '/dict_predictions_deepDMD.pickle', 'rb') as handle:
        dict_run = pickle.load(handle)[RUN_NO]
    if 'observables' in dict_run.keys():
        N_CURVES = len(dict_run.keys()) - 4
    else:
        N_CURVES = len(dict_run.keys())
    ls_train_curves = list(range(int(np.floor(N_CURVES / 3))))
    ls_valid_curves = list(range(ls_train_curves[-1] + 1, ls_train_curves[-1] + 1 + int(np.floor(N_CURVES / 3))))
    ls_test_curves = list(range(ls_valid_curves[-1] + 1, N_CURVES))
    dict_stat = {'train':{},'valid':{},'test':{}}
    num_bas_obs = dict_run[0]['X'].shape[1]
    num_lift_obs = dict_run[0]['psiX'].shape[1] - num_bas_obs
    x_err_train = np.empty(shape=(0,num_bas_obs))
    x_train = np.empty(shape=(0, num_bas_obs))
    psi_err_train = np.empty(shape=(0, num_lift_obs))
    psi_train = np.empty(shape=(0, num_lift_obs))
    for curve in ls_train_curves:
        x_err_train = np.concatenate([x_err_train,dict_run[curve]['X'] - dict_run[curve]['X_one_step']],axis=0)
        x_train = np.concatenate([x_train, dict_run[curve]['X']], axis=0)
        psi_err_train = np.concatenate([psi_err_train, dict_run[curve]['psiX'][:,num_bas_obs:] - dict_run[curve]['psiX_one_step_scaled'][:,num_bas_obs:]], axis=0)
        psi_train = np.concatenate([psi_train, dict_run[curve]['psiX'][:,num_bas_obs:]], axis=0)
    x_err_valid = np.empty(shape=(0, num_bas_obs))
    x_valid = np.empty(shape=(0, num_bas_obs))
    psi_err_valid = np.empty(shape=(0, num_lift_obs))
    psi_valid = np.empty(shape=(0, num_lift_obs))
    for curve in ls_valid_curves:
        x_err_valid = np.concatenate([x_err_valid, dict_run[curve]['X'] - dict_run[curve]['X_one_step']], axis=0)
        x_valid = np.concatenate([x_valid, dict_run[curve]['X']], axis=0)
        psi_err_valid = np.concatenate([psi_err_valid,dict_run[curve]['psiX'][:, num_bas_obs:] - dict_run[curve]['psiX_one_step_scaled'][:, num_bas_obs:]], axis=0)
        psi_valid = np.concatenate([psi_valid, dict_run[curve]['psiX'][:, num_bas_obs:]], axis=0)
    x_err_test = np.empty(shape=(0, num_bas_obs))
    x_test = np.empty(shape=(0, num_bas_obs))
    psi_err_test = np.empty(shape=(0, num_lift_obs))
    psi_test = np.empty(shape=(0, num_lift_obs))
    for curve in ls_test_curves:
        x_err_test = np.concatenate([x_err_test, dict_run[curve]['X'] - dict_run[curve]['X_one_step']], axis=0)
        x_test = np.concatenate([x_test, dict_run[curve]['X']], axis=0)
        psi_err_test = np.concatenate([psi_err_test,dict_run[curve]['psiX'][:, num_bas_obs:] - dict_run[curve]['psiX_one_step_scaled'][:, num_bas_obs:]], axis=0)
        psi_test = np.concatenate([psi_test, dict_run[curve]['psiX'][:, num_bas_obs:]], axis=0)
    for i in range(num_bas_obs):
        dict_stat['train']['x' + str(i+1)] = np.max([0, 100*(1 - np.sum(np.square(x_err_train[:,i]))/np.sum(np.square(x_train[:,i])))])
        dict_stat['valid']['x' + str(i + 1)] = np.max([0, 100 * (1 - np.sum(np.square(x_err_valid[:, i])) / np.sum(np.square(x_valid[:, i])))])
        dict_stat['test']['x' + str(i + 1)] = np.max([0, 100 * (1 - np.sum(np.square(x_err_test[:, i])) / np.sum(np.square(x_test[:, i])))])
    for i in range(num_lift_obs):
        dict_stat['train']['psi' + str(i+1)] = np.max([0, 100*(1 - np.sum(np.square(psi_err_train[:,i]))/np.sum(np.square(psi_train[:,i])))])
        dict_stat['valid']['psi' + str(i + 1)] = np.max([0, 100 * (1 - np.sum(np.square(psi_err_valid[:, i])) / np.sum(np.square(psi_valid[:, i])))])
        dict_stat['test']['psi' + str(i + 1)] = np.max([0, 100 * (1 - np.sum(np.square(psi_err_test[:, i])) / np.sum(np.square(psi_test[:, i])))])
    print('One Step Prediction accuracy of each state and observable:')
    df_stat = pd.DataFrame(dict_stat)
    print(df_stat)
    return df_stat


def plot_fit_XY(dict_run,plot_params,ls_runs,scaled=False,one_step = False):
    n_rows = 7
    n_cols = 6
    graphs_per_run = 2
    f,ax = plt.subplots(n_rows,n_cols,sharex=True,figsize = (plot_params['individual_fig_width']*n_cols,plot_params['individual_fig_height']*n_rows))
    i = 0
    for row_i in range(n_rows):
        for col_i in list(range(0,n_cols)):
            if scaled:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X_scaled'].shape[1]
                for j in range(n_states):
                    ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled'][:, j], '.', color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1))
                    if one_step:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_one_step_scaled'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),
                                              label='x' + str(j + 1) + '[scaled]')
                    else:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_n_step_scaled'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),
                                          label='x' + str(j + 1)+ '[scaled]')
                ax[row_i, col_i].legend()
                try:
                    for j in range(dict_run[ls_runs[i]]['Y_scaled'].shape[1]):
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_scaled'][:, j], '.', color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1))
                        if one_step:
                            ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_scaled_est_one_step'][:, j],color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1), label='y' + str(j + 1) + '[scaled]')
                        else:
                            ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_scaled_est_n_step'][:, j], color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1), label='y' + str(j + 1)+ '[scaled]')
                    ax[row_i, col_i].legend()
                except:
                    print('No output to plot')
            else:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X'].shape[1]
                for j in range(n_states):
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X'][:,j],'.',color = colors[np.mod(j,7)], linewidth=int(j / 7 + 1))
                    if one_step:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_est_one_step'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),
                                              label='x' + str(j + 1))
                    else:
                        ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X_est_n_step'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),label ='x'+str(j+1) )
                try:
                    for j in range(dict_run[ls_runs[i]]['Y'].shape[1]):
                        ax[row_i,col_i].plot(dict_run[ls_runs[i]]['Y'][:,j],'.',color = colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1))
                        if one_step:
                            ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_est_one_step'][:, j],
                                                      color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1), label='y' + str(j + 1))
                        else:
                            ax[row_i,col_i].plot(dict_run[ls_runs[i]]['Y_est_n_step'][:, j], color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1),label ='y'+str(j+1))
                    ax[row_i, col_i].legend()
                except:
                    print('No output to plot')
            i = i+1
            if i == len(ls_runs):
                f.show()
                return f
    f.show()
    return f