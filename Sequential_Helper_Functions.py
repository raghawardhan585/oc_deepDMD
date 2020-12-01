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

def get_error(ls_indices,dict_XY):
    J_error = np.empty(shape=(0,1))
    for i in ls_indices:
        # all_errors = np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step'])
        # all_errors = np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step'])
        all_errors = np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_one_step'])
        # all_errors = np.square(dict_XY[i]['psiX'][:, 0:5] - dict_XY[i]['psiX_est_n_step'][:, 0:5])
        # all_errors = np.append(np.square(dict_XY[i]['X'] - dict_XY[i]['X_est_n_step']) , np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step']))
        # all_errors = np.append(all_errors, np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step']))
        J_error = np.append(J_error, np.mean(all_errors))
    # J_error = np.log10(np.max(J_error))
    J_error = np.mean(J_error)
    return J_error

def get_error_x_and_y(ls_indices,dict_XY):
    J_error = np.empty(shape=(0,1))
    for i in ls_indices:
        # all_errors = np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step'])
        all_errors = np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step'])
        # all_errors = np.square(dict_XY[i]['psiX'][:, 0:5] - dict_XY[i]['psiX_est_n_step'][:, 0:5])
        # all_errors = np.append(np.square(dict_XY[i]['X'] - dict_XY[i]['X_est_n_step']) , np.square(dict_XY[i]['Y'] - dict_XY[i]['Y_est_n_step']))
        # all_errors = np.append(all_errors, np.square(dict_XY[i]['psiX'] - dict_XY[i]['psiX_est_n_step']))
        J_error = np.append(J_error, np.mean(all_errors))
    # J_error = np.log10(np.max(J_error))
    J_error = np.mean(J_error)
    return J_error



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

def get_all_run_info(SYSTEM_NO,RUN_NO,sess):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
    # The Scaler is called upon when required and hence is not required
    # with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_DataScaler.pickle','rb') as handle:
    #     data_Scaler = pickle.load(handle)
    with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_OrderedIndices.pickle','rb') as handle:
        ls_all_indices = pickle.load(handle)
    with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_SimulatedData.pickle','rb') as handle:
        dict_indexed_data = pickle.load(handle) # No scaling appplied here
    # Data used in oc_deepDMD is not required here unless we want to train again
    # with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_ocDeepDMDdata.pickle','rb') as handle:
    #     dict_DATA = pickle.load(handle) # Scaled data
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    try:
        psixpT = tf.get_collection('psixpT')[0]
        psixfT = tf.get_collection('psixfT')[0]
        xpT_feed = tf.get_collection('xpT_feed')[0]
        xfT_feed = tf.get_collection('xfT_feed')[0]
        KxT = tf.get_collection('KxT')[0]
        KxT_num = sess.run(KxT)
        dict_params['psixpT'] = psixpT
        dict_params['psixfT'] = psixfT
        dict_params['xpT_feed'] = xpT_feed
        dict_params['xfT_feed'] =  xfT_feed
        dict_params['KxT_num'] =  KxT_num
    except:
        print('State info not found')
    try:
        ypT_feed = tf.get_collection('ypT_feed')[0]
        yfT_feed = tf.get_collection('yfT_feed')[0]
        dict_params['ypT_feed'] = ypT_feed
        dict_params['yfT_feed'] = yfT_feed
        WhT = tf.get_collection('WhT')[0];
        WhT_num = sess.run(WhT)
        dict_params['WhT_num'] = WhT_num
    except:
        print('No output info found')
    # with open(run_folder_name + '/all_histories.pickle','rb') as handle:
    #     df_train_learning_curves = pd.DataFrame(pickle.load(handle))
    # with open(run_folder_name + '/run_info.pickle','rb') as handle:
    #     df_run_info = pd.DataFrame(pickle.load(handle))
    return dict_params,  ls_all_indices, dict_indexed_data


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

def generate_predictions_pickle_file(SYSTEM_NO,state_only = False, ls_process_runs=[]):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    # Make a predictions folder if one doesn't exist
    if os.path.exists(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle'):
        with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle','rb') as handle:
            dict_predictions_SEQUENTIAL = pickle.load(handle)
    else:
        dict_predictions_SEQUENTIAL = {}
    # Find all available run folders
    ls_all_run_indices = []
    for folder in os.listdir(sys_folder_name+'/Sequential'):
        if folder[0:4] == 'RUN_': # It is a RUN folder
            ls_all_run_indices.append(int(folder[4:]))
    # List of all processed runs are the keys of dict_prediction_SEQUENTIAL
    ls_processed_runs = list(dict_predictions_SEQUENTIAL.keys())
    ls_unprocessed_runs = list(set(ls_all_run_indices) - set(ls_processed_runs))
    # Among the unprocessed runs, only process the specified runs
    if len(ls_process_runs) !=0:
        ls_unprocessed_runs = list(set(ls_unprocessed_runs).intersection(set(ls_process_runs)))
    print('RUNS TO PROCESS - ',ls_unprocessed_runs)
    # Updating the dictionary of predictions
    for run in ls_unprocessed_runs:
        print('RUN: ', run)
        dict_predictions_SEQUENTIAL[run]={}
        sess = tf.InteractiveSession()
        dict_params, _, dict_indexed_data = get_all_run_info(SYSTEM_NO, run, sess)
        try:
            sampling_resolution = 0.01
            dict_psi_phi = oc.observables_and_eigenfunctions(dict_params, sampling_resolution)
            dict_predictions_SEQUENTIAL[run]['X1'] = dict_psi_phi['X1']
            dict_predictions_SEQUENTIAL[run]['X2'] = dict_psi_phi['X2']
            dict_predictions_SEQUENTIAL[run]['observables'] = dict_psi_phi['observables']
            dict_predictions_SEQUENTIAL[run]['eigenfunctions'] = dict_psi_phi['eigenfunctions']
        except:
            print('Cannot find the eigenfunctions and observables as the number of base states is not equal to 2')
        if state_only:
            dict_intermediate = oc.model_prediction_state_only(dict_indexed_data, dict_params, SYSTEM_NO)
        else:
            dict_intermediate = oc.model_prediction(dict_indexed_data, dict_params, SYSTEM_NO)
        for curve_no in dict_intermediate.keys():
            dict_predictions_SEQUENTIAL[run][curve_no] = dict_intermediate[curve_no]
        tf.reset_default_graph()
        sess.close()
    # Saving the dict_predictions folder
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle','wb') as handle:
        pickle.dump(dict_predictions_SEQUENTIAL,handle)
    return


def get_all_run_info_output(SYSTEM_NO,RUN_NO,sess):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
    # The Scaler is called upon when required and hence is not required
    # with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_DataScaler.pickle','rb') as handle:
    #     data_Scaler = pickle.load(handle)
    with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_OrderedIndices.pickle','rb') as handle:
        ls_all_indices = pickle.load(handle)
    with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_SimulatedData.pickle','rb') as handle:
        dict_indexed_data = pickle.load(handle) # No scaling appplied here
    # Data used in oc_deepDMD is not required here unless we want to train again
    # with open(sys_folder_name+'/System_'+str(SYSTEM_NO)+'_ocDeepDMDdata.pickle','rb') as handle:
    #     dict_DATA = pickle.load(handle) # Scaled data
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    try:
        psixfT = tf.get_collection('psixfT')[0]
        xfT_feed = tf.get_collection('xfT_feed')[0]
        yfT_feed = tf.get_collection('yfT_feed')[0]
        WhT = tf.get_collection('WhT')[0];
        WhT_num = sess.run(WhT)
        dict_params['psixfT'] = psixfT
        dict_params['yfT_feed'] = yfT_feed
        dict_params['xfT_feed'] = xfT_feed
        dict_params['WhT_num'] = WhT_num
    except:
        print('No output info found')
    return dict_params,  ls_all_indices, dict_indexed_data

def output_equation_predictions(dict_indexed_data, dict_params, SYSTEM_NUMBER):
    dict_indexed_data_predictions = {}
    for data_index in dict_indexed_data.keys():
        dict_DATA_i = oc.scale_data_using_existing_scaler_folder(dict_indexed_data[data_index], SYSTEM_NUMBER)
        X_scaled = dict_DATA_i['X']
        Y_scaled = dict_DATA_i['Y']
        psiX = dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: X_scaled})
        Y_pred = np.matmul(psiX, dict_params['WhT_num'])
        dict_indexed_data_predictions[data_index] = {}
        dict_indexed_data_predictions[data_index]['Y'] = dict_indexed_data[data_index]['Y']
        dict_indexed_data_predictions[data_index]['Y_pred'] = oc.inverse_transform_Y(Y_pred, SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['Y_scaled'] = Y_scaled
        dict_indexed_data_predictions[data_index]['Y_scaled_pred'] = Y_pred
    return dict_indexed_data_predictions

def generate_predictions_pickle_file_output_only(SYSTEM_NO,ls_process_run_indices):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    # Make a predictions folder if one doesn't exist
    dict_predictions_SEQUENTIAL_OUTPUT = {}
    # Scan all folders to get all Run Indices
    for i in ls_process_run_indices:
        run_folder_name = sys_folder_name+'/Sequential/RUN_' + str(i)
        if os.path.exists(run_folder_name):
            print('RUN: ', i)
            dict_predictions_SEQUENTIAL_OUTPUT[i] = {}
            sess = tf.InteractiveSession()
            dict_params, _, dict_indexed_data = get_all_run_info_output(SYSTEM_NO, i, sess)
            dict_intermediate = output_equation_predictions(dict_indexed_data, dict_params, SYSTEM_NO)
            for curve_no in dict_intermediate.keys():
                dict_predictions_SEQUENTIAL_OUTPUT[i][curve_no] = dict_intermediate[curve_no]
            tf.reset_default_graph()
            sess.close()
        else:
            print('RUN: ', i, ' doesn\'t exist!!!')
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL_OUTPUT.pickle','wb') as handle:
        pickle.dump(dict_predictions_SEQUENTIAL_OUTPUT,handle)
    return

def get_error_output(ls_indices,dict_Y):
    J_error = np.empty(shape=(0,1))
    for i in ls_indices:
        all_errors = np.square(dict_Y[i]['Y'] - dict_Y[i]['Y_pred'])
        J_error = np.append(J_error, np.mean(all_errors))
    # J_error = np.log10(np.max(J_error))
    J_error = np.mean(J_error)
    return J_error

def generate_df_error_output(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    ls_train_curves = list(range(20))
    ls_valid_curves = list(range(20, 40))
    ls_test_curves = list(range(40, 60))
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL_OUTPUT.pickle', 'rb') as handle:
        dict_predictions = pickle.load(handle)
    dict_error = {}
    for run_no in dict_predictions.keys():
        print(run_no)
        dict_error[run_no] = {}
        dict_error[run_no]['train'] = get_error_output(ls_train_curves,dict_predictions[run_no])
        dict_error[run_no]['valid'] = get_error_output(ls_valid_curves, dict_predictions[run_no])
        dict_error[run_no]['test'] = get_error_output(ls_test_curves, dict_predictions[run_no])
    df_error_SEQUENTIAL = pd.DataFrame(dict_error).T
    # Save the file
    with open(sys_folder_name + '/df_error_SEQUENTIAL_OUTPUT.pickle','wb') as handle:
        pickle.dump(df_error_SEQUENTIAL,handle)
    return

def get_prediction_data_output(SYSTEM_NO,RUN_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL_OUTPUT.pickle', 'rb') as handle:
        dict_predictions = pickle.load(handle)
    return dict_predictions[RUN_NO]

def plot_fit_Y(dict_run,plot_params,ls_runs,scaled=False):
    n_rows = 4
    n_cols = 5
    graphs_per_run = 1
    f,ax = plt.subplots(n_rows,n_cols,sharex=True,figsize = (plot_params['individual_fig_width']*n_cols,plot_params['individual_fig_height']*n_rows))
    i = 0
    for row_i in range(n_rows):
        for col_i in list(range(0,n_cols,graphs_per_run)):
            if scaled:
                for j in range(dict_run[ls_runs[i]]['Y_scaled'].shape[1]):
                    ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_scaled'][:, j], '.', color=colors[j])
                    ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_scaled_pred'][:, j], color=colors[j],label='y' + str(j + 1)+ '[scaled]')
                ax[row_i, col_i].legend()
            else:
                for j in range(dict_run[ls_runs[i]]['Y'].shape[1]):
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['Y'][:,j],'.',color = colors[j])
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['Y_pred'][:, j], color=colors[j],label ='y'+str(j+1))
                ax[row_i, col_i].legend()
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

def plot_training_runs_output(SYSTEM_NO,ls_run_no,plot_params):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    # Check that the given list of runs is in the folder
    for run_no in ls_run_no:
        if not os.path.exists(sys_folder_name + '/Sequential/RUN_' + str(run_no)):
            print('[WARNING] RUN_',run_no,' folder does not exist!')
            ls_run_no.remove(run_no)
    # Print the training curves for the runs
    N_runs = len(ls_run_no)
    n_x = int(np.ceil(np.sqrt(N_runs)))
    n_y = int(np.ceil(N_runs/n_x))
    fig = plt.figure(figsize=(plot_params['individual_fig_width'] * n_x, plot_params['individual_fig_height'] * n_y))
    i =1
    for run_no in ls_run_no:
        # Open the run folder
        with open(sys_folder_name + '/Sequential/RUN_' + str(run_no) + '/all_histories.pickle', 'rb') as handle:
            df_run_info = pd.DataFrame(pickle.load(handle)[2])
        ax = fig.add_subplot(n_y, n_x, i)
        i=i+1
        ax.plot(df_run_info.index,np.log10(df_run_info.loc[:,'train MSE']),color = colors[0],label = 'train')
        ax.plot(df_run_info.index, np.log10(df_run_info.loc[:,'valid MSE']), color=colors[1],label = 'valid')
        # ax.legend()
        ax.set_xlabel('# Epochs')
        ax.set_ylabel('Mean squared Error')
        ax.xaxis.label.set_fontsize(plot_params['xy_label_font_size'])
        ax.yaxis.label.set_fontsize(plot_params['xy_label_font_size'])
    fig.show()
    return fig


def generate_df_error(SYSTEM_NO,ls_required_runs):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    ls_train_curves = list(range(20))
    ls_valid_curves = list(range(20, 40))
    ls_test_curves = list(range(40, 60))
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
        dict_predictions_SEQUENTIAL = pickle.load(handle)
    ls_all_available_runs = list(dict_predictions_SEQUENTIAL.keys())
    ls_runs = set(ls_required_runs).intersection(set(ls_all_available_runs))
    dict_error = {}
    for run_no in ls_runs:
        print(run_no)
        dict_error[run_no] = {}
        dict_error[run_no]['train'] = get_error(ls_train_curves,dict_predictions_SEQUENTIAL[run_no])
        dict_error[run_no]['valid'] = get_error(ls_valid_curves, dict_predictions_SEQUENTIAL[run_no])
        dict_error[run_no]['test'] = get_error(ls_test_curves, dict_predictions_SEQUENTIAL[run_no])
    df_error_SEQUENTIAL = pd.DataFrame(dict_error).T
    # Save the file
    if os.path.exists(sys_folder_name + '/df_error_SEQUENTIAL.pickle'):
        ip = input('Do you wanna write over the df_error file[y/n]?')
        if ip == 'y':
            os.remove(sys_folder_name + '/df_error_SEQUENTIAL.pickle')
            with open(sys_folder_name + '/df_error_SEQUENTIAL.pickle', 'wb') as handle:
                pickle.dump(df_error_SEQUENTIAL, handle)
    else:
        with open(sys_folder_name + '/df_error_SEQUENTIAL.pickle','wb') as handle:
            pickle.dump(df_error_SEQUENTIAL,handle)
    return

def generate_df_error_x_and_y(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    ls_train_curves = list(range(20))
    ls_valid_curves = list(range(20, 40))
    ls_test_curves = list(range(40, 60))
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
        dict_predictions_SEQUENTIAL = pickle.load(handle)
    dict_error = {}
    for run_no in dict_predictions_SEQUENTIAL.keys():
        print(run_no)
        dict_error[run_no] = {}
        dict_error[run_no]['train'] = get_error_x_and_y(ls_train_curves,dict_predictions_SEQUENTIAL[run_no])
        dict_error[run_no]['valid'] = get_error_x_and_y(ls_valid_curves, dict_predictions_SEQUENTIAL[run_no])
        dict_error[run_no]['test'] = get_error_x_and_y(ls_test_curves, dict_predictions_SEQUENTIAL[run_no])
    df_error_SEQUENTIAL = pd.DataFrame(dict_error).T
    # Save the file
    if os.path.exists(sys_folder_name + '/df_error_SEQUENTIAL_x_and_y.pickle'):
        ip = input('Do you wanna write over the df_error file[y/n]?')
        if ip == 'y':
            os.remove(sys_folder_name + '/df_error_SEQUENTIAL_x_and_y.pickle')
            with open(sys_folder_name + '/df_error_SEQUENTIAL_x_and_y.pickle', 'wb') as handle:
                pickle.dump(df_error_SEQUENTIAL, handle)
    else:
        with open(sys_folder_name + '/df_error_SEQUENTIAL_x_and_y.pickle','wb') as handle:
            pickle.dump(df_error_SEQUENTIAL,handle)
    return

def get_prediction_data(SYSTEM_NO,RUN_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    with open(sys_folder_name + '/dict_predictions_SEQUENTIAL.pickle', 'rb') as handle:
        dict_predictions = pickle.load(handle)
    return dict_predictions[RUN_NO]

def plot_fit_XY(dict_run,plot_params,ls_runs,scaled=False,observables=False,one_step = False):
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
                    if one_step:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled_est_one_step'][:, j], color=colors[j],
                                              label='x' + str(j + 1) + '[scaled]')
                    else:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled_est_n_step'][:, j], color=colors[j],
                                          label='x' + str(j + 1)+ '[scaled]')
                ax[row_i, col_i].legend()
                try:
                    for j in range(dict_run[ls_runs[i]]['Y_scaled'].shape[1]):
                        ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled'][:, j], '.', color=colors[n_states + j])
                        if one_step:
                            ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled_est_one_step'][:, j],color=colors[n_states + j],
                                                      label='y' + str(j + 1) + '[scaled]')
                        else:
                            ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_scaled_est_n_step'][:, j], color=colors[n_states + j],
                                                  label='y' + str(j + 1)+ '[scaled]')
                    ax[row_i, col_i + 1].legend()
                except:
                    print('No output to plot')
            else:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X'].shape[1]
                for j in range(n_states):
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X'][:,j],'.',color = colors[j])
                    if one_step:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_est_one_step'][:, j], color=colors[j],
                                              label='x' + str(j + 1))
                    else:
                        ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X_est_n_step'][:, j], color=colors[j],label ='x'+str(j+1) )
                ax[row_i, col_i].legend()
                try:
                    for j in range(dict_run[ls_runs[i]]['Y'].shape[1]):
                        ax[row_i,col_i+1].plot(dict_run[ls_runs[i]]['Y'][:,j],'.',color = colors[n_states+j])
                        if one_step:
                            ax[row_i, col_i + 1].plot(dict_run[ls_runs[i]]['Y_est_one_step'][:, j],
                                                      color=colors[n_states + j], label='y' + str(j + 1))
                        else:
                            ax[row_i,col_i+1].plot(dict_run[ls_runs[i]]['Y_est_n_step'][:, j], color=colors[n_states+j],label ='y'+str(j+1))
                    ax[row_i, col_i+1].legend()
                except:
                    print('No output to plot')
            if observables:
                # Plot the observables
                for j in range(n_states,dict_run[ls_runs[i]]['psiX'].shape[1]):
                    ax[row_i,col_i+2].plot(dict_run[ls_runs[i]]['psiX'][:,j],'.',color = colors[np.mod(j,7)],linewidth = int(j/7+1))
                    if one_step:
                        ax[row_i, col_i + 2].plot(dict_run[ls_runs[i]]['psiX_est_one_step'][:, j],
                                                  color=colors[np.mod(j, 7)], linewidth=int(j / 7 + 1),label='x_' + str(j + 1))
                    else:
                        ax[row_i,col_i+2].plot(dict_run[ls_runs[i]]['psiX_est_n_step'][:, j], color=colors[np.mod(j,7)],linewidth = int(j/7+1),label ='x_'+str(j+1) )
                ax[row_i, col_i+2].legend()
            i = i+1
            if i == len(ls_runs):
                break
    f.show()
    return f