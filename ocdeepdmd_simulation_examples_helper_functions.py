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

# Help to store data for oc_deepDMD
def sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data,SYSTEM_NO):
    # Sorting into deep DMD format
    ls_all_indices = np.arange(int(np.ceil(2 / 3 * N_CURVES)))  # We take 2/3rd of the data - The training and validation set
    # random.shuffle(ls_all_indices) # Not required as the initial conditions are already shuffled
    print('[INFO]: Shape of Y : ',dict_indexed_data[0]['Y'].shape)
    n_data_pts = dict_indexed_data[0]['X'].shape[0]
    Xp = np.empty((0, dict_indexed_data[0]['X'].shape[1]))
    Xf = np.empty((0, dict_indexed_data[0]['X'].shape[1]))
    Yp = np.empty((0, dict_indexed_data[0]['Y'].shape[1]))
    Yf = np.empty((0, dict_indexed_data[0]['Y'].shape[1]))
    for i in ls_all_indices:
        Xp = np.concatenate([Xp, dict_indexed_data[i]['X'][0:-1, :]], axis=0)
        Xf = np.concatenate([Xf, dict_indexed_data[i]['X'][1:, :]], axis=0)
        Yp = np.concatenate([Yp, dict_indexed_data[i]['Y'][0:-1, :]], axis=0)
        Yf = np.concatenate([Yf, dict_indexed_data[i]['Y'][1:, :]], axis=0)
    dict_DATA_RAW = {'Xp': Xp, 'Xf': Xf, 'Yp': Yp, 'Yf': Yf}
    n_train = int(np.ceil(len(dict_DATA_RAW['Xp']) / 2))  # Segregate half of data as training
    dict_DATA_TRAIN_RAW = {'Xp': dict_DATA_RAW['Xp'][0:n_train], 'Xf': dict_DATA_RAW['Xf'][0:n_train],
                           'Yp': dict_DATA_RAW['Yp'][0:n_train], 'Yf': dict_DATA_RAW['Yf'][0:n_train]}
    # _, dict_Scaler, _ = scale_train_data(dict_DATA_TRAIN_RAW, 'min max')
    _, dict_Scaler, _ = scale_train_data(dict_DATA_TRAIN_RAW, 'standard')
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
        pickle.dump(dict_Scaler, handle)
    dict_DATA = scale_data_using_existing_scaler_folder(dict_DATA_RAW, SYSTEM_NO)
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
        pickle.dump(dict_DATA, handle)
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'wb') as handle:
        pickle.dump(dict_indexed_data, handle)
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle', 'wb') as handle:
        pickle.dump(ls_all_indices, handle)  # Only training and validation indices are stored
    # Store the data in Koopman
    with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle',
              'wb') as handle:
        pickle.dump(dict_DATA, handle)
    return
def sort_to_DMD_folder_multi_step(storage_folder, N_CURVES, dict_indexed_data,ls_prediction_steps,SYSTEM_NO):
    # Sorting into deep DMD format
    ls_all_indices = np.arange(int(np.ceil(2 / 3 * N_CURVES)))  # We take 2/3rd of the data - The training and validation set
    # random.shuffle(ls_all_indices) # Not required as the initial conditions are already shuffled
    if 1 not in ls_prediction_steps:
        ls_prediction_steps.append(1)
    ls_prediction_steps.sort()
    print('[INFO]: Shape of Y : ',dict_indexed_data[0]['Y'].shape)
    n_data_pts = dict_indexed_data[0]['X'].shape[0]
    Xp = {}
    Xf = {}
    for step in ls_prediction_steps:
        if step <= n_data_pts:
            Xp[step] = np.empty((0, dict_indexed_data[0]['X'].shape[1]))
            Xf[step] = np.empty((0, dict_indexed_data[0]['X'].shape[1]))
            for i in ls_all_indices:
                Xp[step] = np.concatenate([Xp[step], dict_indexed_data[i]['X'][0:-step, :]], axis=0)
                Xf[step] = np.concatenate([Xf[step], dict_indexed_data[i]['X'][step:, :]], axis=0)
        else:
            print('[WARNING] Predictive step ',step,' cannot be processed due to insufficient data')
    # Xp = np.empty((0, dict_indexed_data[0]['X'].shape[1]))
    # Xf = np.empty((0, dict_indexed_data[0]['X'].shape[1]))
    Yp = np.empty((0, dict_indexed_data[0]['Y'].shape[1]))
    Yf = np.empty((0, dict_indexed_data[0]['Y'].shape[1]))
    for i in ls_all_indices:
        # Xp = np.concatenate([Xp, dict_indexed_data[i]['X'][0:-1, :]], axis=0)
        # Xf = np.concatenate([Xf, dict_indexed_data[i]['X'][1:, :]], axis=0)
        Yp = np.concatenate([Yp, dict_indexed_data[i]['Y'][0:-1, :]], axis=0)
        Yf = np.concatenate([Yf, dict_indexed_data[i]['Y'][1:, :]], axis=0)
    dict_DATA_RAW = {'Xp': Xp, 'Xf': Xf, 'Yp': Yp, 'Yf': Yf}
    n_train = int(np.ceil(len(dict_DATA_RAW['Xp']) / 2))  # Segregate half of data as training
    dict_DATA_TRAIN_RAW = {'Xp': dict_DATA_RAW['Xp'][1][0:n_train], 'Xf': dict_DATA_RAW['Xf'][1][0:n_train],
                           'Yp': dict_DATA_RAW['Yp'][0:n_train], 'Yf': dict_DATA_RAW['Yf'][0:n_train]}
    # _, dict_Scaler, _ = scale_train_data(dict_DATA_TRAIN_RAW, 'min max')
    _, dict_Scaler, _ = scale_train_data(dict_DATA_TRAIN_RAW, 'standard')
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
        pickle.dump(dict_Scaler, handle)
    dict_DATA = scale_data_using_existing_scaler_folder(dict_DATA_RAW, SYSTEM_NO)
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
        pickle.dump(dict_DATA, handle)
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle', 'wb') as handle:
        pickle.dump(dict_indexed_data, handle)
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle', 'wb') as handle:
        pickle.dump(ls_all_indices, handle)  # Only training and validation indices are stored
    # Store the data in Koopman
    with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle',
              'wb') as handle:
        pickle.dump(dict_DATA, handle)
    return

def write_bash_script(DEVICE_TO_RUN_ON,dict_run_conditions,SYSTEM_NO,NO_OF_ITERATIONS_PER_GPU,NO_OF_ITERATIONS_IN_CPU):
    with open('/Users/shara/Desktop/oc_deepDMD/' + str(DEVICE_TO_RUN_ON) + '_run.sh', 'w') as bash_exec:
        bash_exec.write('#!/bin/bash \n')
        bash_exec.write('rm -rf _current_run_saved_files \n')
        bash_exec.write('mkdir _current_run_saved_files \n')
        bash_exec.write('rm -rf Run_info \n')
        bash_exec.write('mkdir Run_info \n')
        bash_exec.write(
            '# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] \n')
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
                    general_run = 'python3 gen_control_Koopman_SHARA_addition.py \'/gpu:' + str(i) + '\' ' + str(
                        SYSTEM_NO) + ' 0 1 0 '
                    run_params = str(RUN_NO)
                    for items in dict_run_conditions[i].keys():
                        run_params = run_params + ' ' + str(dict_run_conditions[i][items])
                    write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NO) + '.txt &\n'
                    bash_exec.write(general_run + run_params + write_to_file)
                    RUN_NO = RUN_NO + 1
            elif i in ls_cpu:
                for j in range(NO_OF_ITERATIONS_IN_CPU):
                    general_run = 'python3 gen_control_Koopman_SHARA_addition.py \'/cpu:0\' ' + str(
                        SYSTEM_NO) + ' 0 1 0 '
                    run_params = str(RUN_NO)
                    for items in dict_run_conditions[i].keys():
                        run_params = run_params + ' ' + str(dict_run_conditions[i][items])
                    write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NO) + '.txt &\n'
                    bash_exec.write(general_run + run_params + write_to_file)
                    RUN_NO = RUN_NO + 1
        bash_exec.write('echo "Running all sessions" \n')
        bash_exec.write('wait \n')
        bash_exec.write('echo "All sessions are complete" \n')
        bash_exec.write('echo "=======================================================" \n')
        # bash_exec.write('cd .. \n')
        # bash_exec.write('rm -R _current_run_saved_files ')
        # bash_exec.write('rm -R Run_info ')
        # bash_exec.write('cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files ')
        # bash_exec.write('cp -a oc_deepDMD/Run_info/ Run_info  ')
        # bash_exec.write('cd oc_deepDMD/  ')
        # cp -a _current_run_saved_files/ oc_deepDMD/_current_run_saved_files
        # cp -a Run_info/ oc_deepDMD/Run_info
    return


def get_all_run_info(SYSTEM_NO,RUN_NO,sess):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    run_folder_name = sys_folder_name + '/RUN_' + str(RUN_NO)
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
        ypT_feed = tf.get_collection('ypT_feed')[0]
        yfT_feed = tf.get_collection('yfT_feed')[0]
        KxT = tf.get_collection('KxT')[0]
        KxT_num = sess.run(KxT)
        dict_params['psixpT'] = psixpT
        dict_params['psixfT'] = psixfT
        dict_params['xpT_feed'] = xpT_feed
        dict_params['xfT_feed'] =  xfT_feed
        dict_params['ypT_feed'] = ypT_feed
        dict_params['yfT_feed'] = yfT_feed
        dict_params['KxT_num'] =  KxT_num
    except:
        print('State info not found')
    try:
        WhT = tf.get_collection('WhT')[0];
        WhT_num = sess.run(WhT)
        dict_params['WhT_num'] = WhT_num
    except:
        print('No output info found')
    # ---------
    # #USELESS - Only contains the weights and biases info if we want to retrain the neural network
    # with open(run_folder_name + '/constrainedNN-Model.pickle','rb') as handle:
    #     dict_ALL= pickle.load(handle)
    # ---------
    with open(run_folder_name + '/all_histories.pickle','rb') as handle:
        df_train_learning_curves = pd.DataFrame(pickle.load(handle))
    with open(run_folder_name + '/run_info.pickle','rb') as handle:
        df_run_info = pd.DataFrame(pickle.load(handle))
    return dict_params,  ls_all_indices, dict_indexed_data,  df_train_learning_curves,df_run_info #, data_Scaler, dict_DATA
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
    destination_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NUMBER)
    current_run_no = -1
    for items in os.listdir(destination_folder):
        if items[0:4] == 'RUN_':
            current_run_no = np.max([current_run_no, int(items[4:])])
    current_run_no = current_run_no + 1
    # Transfer the files
    for items in list(set(os.listdir(source_folder)) - {'.DS_Store'}):
        shutil.move(source_folder + '/' + items, destination_folder + '/RUN_' + str(current_run_no))
        shutil.move(runinfo_folder + '/' + items + '.txt',
                    destination_folder + '/RUN_' + str(current_run_no) + '/RUN_' + str(current_run_no) + '.txt')
        current_run_no = current_run_no + 1
    return


def plot_training_runs(SYSTEM_NO,ls_run_no,plot_params):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    # Check that the given list of runs is in the folder
    for run_no in ls_run_no:
        if not os.path.exists(sys_folder_name + '/RUN_' + str(run_no)):
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
        with open(sys_folder_name + '/RUN_' + str(run_no) + '/all_histories.pickle', 'rb') as handle:
            df_run_info = pd.DataFrame(pickle.load(handle))
        ax = fig.add_subplot(n_y, n_x, i)
        i=i+1
        ax.plot(df_run_info.index*1000,np.log10(df_run_info.loc[:,'x train MSE']),color = colors[0],label = 'x')
        ax.plot(df_run_info.index * 1000, np.log10(df_run_info.loc[:,'y train MSE']), color=colors[1],label = 'y')
        # ax.legend()
        ax.set_xlabel('# Epochs')
        ax.set_ylabel('Mean squared Error')
        ax.xaxis.label.set_fontsize(plot_params['xy_label_font_size'])
        ax.yaxis.label.set_fontsize(plot_params['xy_label_font_size'])
    fig.show()
    return fig
# ----------------------------------------------------------------------------------------------------------------
# SCALING FUNCTIONS
# ----------------------------------------------------------------------------------------------------------------
def scale_train_data(dict_DATA_IN,method ='standard',WITH_MEAN_FOR_STANDARD_SCALER_X = True, WITH_MEAN_FOR_STANDARD_SCALER_Y = True):
    dict_DATA_OUT = {}
    dict_Scaler = {}
    dict_transform_matrices = {}
    if method not in ['standard','min max','normalizer']:
        print('Error in the method name specified')
        print('Taking the standard method as the default')
        method = 'standard'
    X_all = np.append(dict_DATA_IN['Xp'],dict_DATA_IN['Xf'],axis=0)
    X_n_vars = X_all.shape[1]
    X_PT = np.zeros(shape = (X_n_vars,X_n_vars))
    X_bT = np.zeros(shape = (1,X_n_vars))
    if method == 'standard':
        X_Scale = StandardScaler(with_mean = WITH_MEAN_FOR_STANDARD_SCALER_X).fit(X_all)
        X_mean = X_all.mean(axis=0)
        X_std = X_all.std(axis=0)
        for i in range(X_n_vars):
            X_PT[i, i] = 1 / (X_std[i])
            X_bT[0, i] = -X_mean[i] / (X_std[i])
    elif method == 'min max':
        X_Scale = MinMaxScaler(feature_range=(0,1)).fit(X_all)
        X_max = X_all.max(axis=0)
        X_min = X_all.min(axis=0)
        for i in range(X_n_vars):
            X_PT[i,i] = 1/(X_max[i]-X_min[i])
            X_bT[0, i] = -X_min[i] / (X_max[i] - X_min[i])
    elif method == 'normalizer':
        X_Scale  = Normalizer().fit(np.append(dict_DATA_IN['Xp'],dict_DATA_IN['Xf'],axis=0))
    dict_Scaler['X Scale'] = X_Scale
    dict_DATA_OUT['Xp'] = X_Scale.transform(dict_DATA_IN['Xp'])
    dict_DATA_OUT['Xf'] = X_Scale.transform(dict_DATA_IN['Xf'])
    try:
        dict_transform_matrices['X_PT'] = X_PT
        dict_transform_matrices['X_bT'] = X_bT
    except:
        print('[WARNING]: State did not identify scaling matrices')
    try:
        Y_all = np.append(dict_DATA_IN['Yp'],dict_DATA_IN['Yf'],axis=0)
        Y_n_vars = Y_all.shape[1]
        Y_PT = np.zeros(shape=(Y_n_vars, Y_n_vars))
        Y_bT = np.zeros(shape=(1, Y_n_vars))
        if method == 'standard':
            Y_Scale= StandardScaler(with_mean = WITH_MEAN_FOR_STANDARD_SCALER_Y).fit(Y_all)
            Y_mean = Y_all.mean(axis=0)
            Y_std = Y_all.std(axis=0)
            for i in range(Y_n_vars):
                Y_PT[i, i] = 1 / (Y_std[i])
                Y_bT[0, i] = -Y_mean[i] / (Y_std[i])
        elif method == 'min max':
            Y_Scale = MinMaxScaler(feature_range=(0,1)).fit(Y_all)
            Y_max = Y_all.max(axis=0)
            Y_min = Y_all.min(axis=0)
            for i in range(Y_n_vars):
                Y_PT[i, i] = 1 / (Y_max[i] - Y_min[i])
                Y_bT[0, i] = -Y_min[i] / (Y_max[i] - Y_min[i])
        elif method == 'normalizer':
            Y_Scale = Normalizer().fit(Y_all)
        dict_Scaler['Y Scale'] = Y_Scale
        dict_DATA_OUT['Yp'] = Y_Scale.transform(Y_all)
        dict_DATA_OUT['Yf'] = Y_Scale.transform(Y_all)
        try:
            dict_transform_matrices['Y_PT'] = Y_PT
            dict_transform_matrices['Y_bT'] = Y_bT
        except:
            print('[WARNING]: Output did not identify scaling matrices')
    except:
        print('[WARNING] No output provided')
    return dict_DATA_OUT,dict_Scaler,dict_transform_matrices

def inverse_transform_X(X_in,SYSTEM_NUMBER):
    with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_'+ str(SYSTEM_NUMBER) +'/System_' + str(SYSTEM_NUMBER) + '_DataScaler.pickle', 'rb') as handle:
        TheScaler = pickle.load(handle)
    if 'X Scale' in TheScaler.keys():
        if type(X_in) == dict:
            X_out = {}
            for step in X_in.keys():
                X_out[step] = TheScaler['X Scale'].transform(X_in[step])
        else:
            X_out = TheScaler['X Scale'].inverse_transform(X_in)
    else:
        X_out = X_in
    return X_out

def inverse_transform_Y(Y_in,SYSTEM_NUMBER):
    with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_'+ str(SYSTEM_NUMBER) +'/System_' + str(SYSTEM_NUMBER) + '_DataScaler.pickle', 'rb') as handle:
        TheScaler = pickle.load(handle)
    if 'Y Scale' in TheScaler.keys():
        Y_out = TheScaler['Y Scale'].inverse_transform(Y_in)
    else:
        Y_out = Y_in
    return Y_out

def scale_data_using_existing_scaler_folder(dict_DATA_IN,SYSTEM_NUMBER):
    dict_DATA_OUT = {}
    with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_'+ str(SYSTEM_NUMBER) +'/System_' + str(SYSTEM_NUMBER) + '_DataScaler.pickle', 'rb') as handle:
        TheScaler = pickle.load(handle)
    for item in dict_DATA_IN.keys():
        if item in ['Xp','Xf','X']:
            if 'X Scale' in TheScaler.keys():
                if type(dict_DATA_IN[item]) == dict:
                    dict_DATA_OUT[item] = {}
                    for step in dict_DATA_IN[item].keys():
                        dict_DATA_OUT[item][step] = TheScaler['X Scale'].transform(dict_DATA_IN[item][step])
                else:
                    dict_DATA_OUT[item] = TheScaler['X Scale'].transform(dict_DATA_IN[item])
            else:
                dict_DATA_OUT[item] = dict_DATA_IN[item]
        elif item in ['Yp', 'Yf','Y']:
            if 'Y Scale' in TheScaler.keys():
                dict_DATA_OUT[item] = TheScaler['Y Scale'].transform(dict_DATA_IN[item])
            else:
                dict_DATA_OUT[item] = dict_DATA_IN[item]
    return dict_DATA_OUT

# ----------------------------------------------------------------------------------------------------------------
# SYSTEMS FOR DATA GENERATION
## ----------------------------------------------------------------------------------------------------------------
def plot_training_valid_test_states(SYSTEM_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle','rb') as handle:
        dict_DATA = pickle.load(handle)
    n_states = len(dict_DATA[0]['X'][0])
    n_outputs = len(dict_DATA[0]['Y'][0])
    n_curves = len(dict_DATA)
    N_curves_split = int(np.ceil(n_curves/3))
    f,ax = plt.subplots(n_states+n_outputs,3,sharex=True,figsize = (3*2,2*(n_states+n_outputs)))
    for i in range(0,N_curves_split):
        for j in range(n_states):
            ax[j, 0].plot(dict_DATA[i]['X'][:, j], color=colors[np.mod(j,7)],label='X'+str(j))
        for j in range(n_outputs):
            ax[n_states+j,0].plot(dict_DATA[i]['Y'][:, j], color=colors[np.mod(n_states+j,7)],label='Y'+str(j))
        # ax[0,0].plot(dict_DATA[i]['X'][:,0],color=colors[0])
        # ax[1,0].plot(dict_DATA[i]['X'][:, 1], color=colors[1])
    ax[0,0].set_title('Training Data')
    for i in range(N_curves_split,2*N_curves_split):
        for j in range(n_states):
            ax[j, 1].plot(dict_DATA[i]['X'][:, j],'.', color=colors[np.mod(j, 7)],label='X'+str(j))
        for j in range(n_outputs):
            ax[n_states + j, 1].plot(dict_DATA[i]['Y'][:, j],'.', color=colors[np.mod(n_states + j, 7)],
                                     label='Y' + str(j))
        # ax[0,1].plot(dict_DATA[i]['X'][:,0],color=colors[0])
        # ax[1,1].plot(dict_DATA[i]['X'][:, 1], color=colors[1])
    ax[0,1].set_title('Validation Data')
    for i in range(2*N_curves_split,n_curves):
        for j in range(n_states):
            ax[j, 2].plot(dict_DATA[i]['X'][:, j],'.', color=colors[np.mod(j, 7)],label='X'+str(j))
        for j in range(n_outputs):
            ax[n_states + j, 2].plot(dict_DATA[i]['Y'][:, j],'.', color=colors[np.mod(n_states + j, 7)],
                                     label='Y' + str(j))
        # ax[0,2].plot(dict_DATA[i]['X'][:,0],color=colors[0])
        # ax[1,2].plot(dict_DATA[i]['X'][:, 1], color=colors[1])
    ax[0,2].set_title('Test Data')
    f.show()
    return

def plot_one_curve(SYSTEM_NO,CURVE_NO):
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle','rb') as handle:
        dict_DATA = pickle.load(handle)[CURVE_NO]
    n_states = len(dict_DATA['X'][0])
    n_outputs = len(dict_DATA['Y'][0])
    f,ax = plt.subplots(2,1,figsize=(3,10))
    for j in range(n_states):
        ax[0].plot(dict_DATA['X'][:, j], '.', color=colors[np.mod(j,7)],label='X'+str(j),linewidth = 1 + np.int(j/7))
    for j in range(n_outputs):
        ax[1].plot(dict_DATA['Y'][:, j], '.', color=colors[np.mod(n_states+j,7)],label='Y'+str(j))
        # ax[0,0].plot(dict_DATA[i]['X'][:,0],color=colors[0])
        # ax[1,0].plot(dict_DATA[i]['X'][:, 1], color=colors[1])
    ax[0].legend()
    ax[1].legend()
    ax[0].set_title('States')
    ax[1].set_title('Outputs')
    f.show()
    return

def generate_2state_initial_condition():
    r = np.random.uniform(5.,10.)
    theta = np.random.uniform(0,2*np.pi)
    return np.array([[r*np.cos(theta),r*np.sin(theta)]])

def sim_sys_1_2(sys_params):
    if sys_params['x0'].shape != (1,2):
        print('[ERROR]: Incorrect dimensions of the initial condition x0. It should be array of (1,2)')
        exit(1)
    X = sys_params['x0']
    for i in range(sys_params['N_data_points']-1):
        # Autonomous Linear Components0
        x1_next = sys_params['A'][0,0] * X[-1, 0] + sys_params['A'][0,1] * X[-1, 1] #+ sys_params['delta']* U[i,0]**2
        x2_next = sys_params['A'][1,0] * X[-1, 0] + sys_params['A'][1,1] * X[-1,1] #+ sys_params['epsilon']*X[-1,1]**2*U[i,0]
        # Nonlinear dynamics components
        x2_next = x2_next + sys_params['gamma'] * X[-1,0]**2
        X = np.concatenate([X,np.array([[x1_next,x2_next]])],axis=0)
    Y = X[:,0:1] * X[:,1:2]
    dict_DATA = {'X': X, 'Y': Y}
    return dict_DATA

def data_gen_sys_1_2(sys_params, N_CURVES,SYSTEM_NO):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        get_input = input('Do you wanna delete the existing system[y/n]? ')
        if get_input == 'y':
            shutil.rmtree(storage_folder)
            os.mkdir(storage_folder)
        else:
            return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data = {}
    f,ax = plt.subplots(2,1)
    X0 = np.empty(shape=(0,2))
    for i in range(N_CURVES):
        sys_params['x0'] = generate_2state_initial_condition()
        X0 = np.concatenate([X0,sys_params['x0']],axis=0)
        dict_indexed_data[i]= sim_sys_1_2(sys_params)
        ax[0].plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])
        ax[1].plot(dict_indexed_data[i]['Y'])
    ax[0].plot(X0[:,0],X0[:,1],'*')
    plt.show()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data, SYSTEM_NO)
    # # Sorting into deep DMD format
    # Xp = np.empty((0,2))
    # Xf = np.empty((0,2))
    # Yp = np.empty((0,1))
    # Yf = np.empty((0,1))
    # ls_all_indices = np.arange(int(np.ceil(2/3*N_CURVES))) # We take 2/3rd of the data - The training and validation set
    # # random.shuffle(ls_all_indices) # Not required as the initial conditions are already shuffled
    # for i in ls_all_indices:
    #     Xp = np.concatenate([Xp,dict_indexed_data[i]['X'][0:-1,:]],axis=0)
    #     Xf = np.concatenate([Xf,dict_indexed_data[i]['X'][1:,:]],axis=0)
    #     Yp = np.concatenate([Yp,dict_indexed_data[i]['Y'][0:-1,:]],axis=0)
    #     Yf = np.concatenate([Yf,dict_indexed_data[i]['Y'][1:,:]],axis=0)
    # dict_DATA_RAW = {'Xp': Xp, 'Xf': Xf, 'Yp': Yp, 'Yf': Yf}
    # n_train = int(np.ceil(len(dict_DATA_RAW['Xp'])/2)) # Segregate half of data as training
    # dict_DATA_TRAIN_RAW = {'Xp': dict_DATA_RAW['Xp'][0:n_train], 'Xf': dict_DATA_RAW['Xf'][0:n_train], 'Yp': dict_DATA_RAW['Yp'][0:n_train], 'Yf': dict_DATA_RAW['Yf'][0:n_train]}
    # _,dict_Scaler,_ = scale_train_data(dict_DATA_TRAIN_RAW,'min max')
    # with open(storage_folder + '/System_'+ str(SYSTEM_NO) +'_DataScaler.pickle', 'wb') as handle:
    #     pickle.dump(dict_Scaler,handle)
    # dict_DATA = scale_data_using_existing_scaler_folder(dict_DATA_RAW,SYSTEM_NO)
    # with open(storage_folder + '/System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
    #     pickle.dump(dict_DATA,handle)
    # with open(storage_folder + '/System_'+ str(SYSTEM_NO) + '_SimulatedData.pickle', 'wb') as handle:
    #     pickle.dump(dict_indexed_data,handle)
    # with open(storage_folder + '/System_'+ str(SYSTEM_NO) +'_OrderedIndices.pickle', 'wb') as handle:
    #     pickle.dump(ls_all_indices,handle) # Only training and validation indices are stored
    # # Store the data in Koopman
    # with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
    #     pickle.dump(dict_DATA,handle)
    return

# ================================================
# SYSTEM - Incoherent Feed forward loop with a reporter as output
# ================================================

def iffl_system(x,t,gamma_1,gamma_2,k_1,k_2n,k_2d):
    x1dot = - gamma_1 * x[0] + k_1
    x2dot = - gamma_2 * x[1] + k_2n/(k_2d + x[0])
    return np.array([x1dot,x2dot])

# def gen_data_iffl_system():


# ================================================
# SYSTEM - Activator Repressor Clock with 4states [arc4s]
# ================================================
def activator_repressor_clock_4states(x,t,gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m):
    x1dot = - delta_A * x[0] + (alpha_A*(x[1]/K_A)**n + alpha_A0)/(1 + (x[1]/K_A)**n + (x[3]/K_B)**m)
    x2dot = - gamma_A * x[1] + kappa_A*x[0]
    x3dot = - delta_B * x[2] + (alpha_B*(x[1]/K_A)**n + alpha_B0)/(1 + (x[1]/K_A)**n)
    x4dot = - gamma_B * x[3] + kappa_B*x[2]
    return np.array([x1dot,x2dot,x3dot,x4dot])

def activator_repressor_clock_2states(x,t,gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m):
    x1dot = - gamma_A * x[0] + kappa_A/delta_A * (alpha_A*(x[0]/K_A)**n + alpha_A0)/(1 + (x[0]/K_A)**n + (x[1]/K_B)**m)
    x2dot = - gamma_B * x[1] + kappa_B/delta_B*(alpha_B*(x[0]/K_A)**n + alpha_B0)/(1 + (x[0]/K_A)**n)
    return np.array([x1dot,x2dot])

def duffing_oscillator(x,t,alpha,beta,delta):
    x1dot = x[1]
    x2dot = - delta*x[1] - x[0] *(beta + alpha*(x[0]**2))
    return np.array([x1dot,x2dot])

def MEMS_accelerometer(x,t,m,k,c,k3):
    x1dot = x[1]
    x2dot = - k/m*x[0] - c/m*x[1]  - k3/m*x[0]**3
    return np.array([x1dot,x2dot])

def incoherent_feedforward_loop(x,t,k1,gamma1,k2n,k2d,gamma2):
    x1dot = k1 - gamma1*x[0]
    x2dot = - gamma2 * x[1] + k2n/(k2d+x[0])
    return np.array([x1dot,x2dot])

def combinatorial_promoter(x,t,k1f,k1r,k2f,k2r,k3f,k3r,k4f,k4r,k5f,k5r,k6f,k6r,k7f,k7r,k8f,delta,u1,u2):
    xdot = np.zeros(len(x))
    xdot[0] = -k1f*u1*x[0] + k1r*x[4];
    xdot[1] = -k2f*u2*x[1] + k2r*x[5] -k4f*x[1]*x[2] + k4r*x[7] - k5f*x[1]*x[6] + k5r*x[8] #+ 0.2*x[10];
    xdot[2] = -k3f * x[2] * x[4] + k3r * x[6] - k4f * x[1] * x[2] + k4r * x[7];
    xdot[3] = - k7f * x[3] * x[6] + k7r * x[9] + k8f * x[9] ;
    xdot[4] = k1f * u1 * x[0]  - k1r * x[4] - k3f * x[2] * x[4] + k3r * x[6] - k6f * x[4] * x[7] + k6r * x[8];
    xdot[5] = k2f * u2 * x[1]- k2r*x[5] # - 0.2*x[3];
    xdot[6] = k3f*x[2]*x[4] - k3r*x[6] - k5f*x[1]*x[6] + k5r*x[8] - k7f*x[3]*x[6] + k7r*x[9] + k8f*x[9];
    xdot[7] = k4f * x[1] * x[2]- k4r * x[7]  -k6f * x[4] * x[7] + k6r * x[8] ;
    xdot[8] = k5f*x[1]*x[6] - k5r*x[8] + k6f*x[4]*x[7] - k6r*x[8];
    xdot[9] = k7f * x[3] * x[6] - k7r * x[9] -  k8f * x[9];
    xdot[10] = k8f*x[9] - delta*x[10];
    return xdot

def glycolytic_oscillator(x,t,k1,k2,k3,k4,k5,k6,k7,K1,kappa,mu,q,J0,N,A):
    xdot = np.zeros(len(x))
    xdot[0] = J0 - k1*x[0]*x[5]/(1 +(x[5]/K1)**q)
    xdot[1] = 2*k1*x[0]*x[5]/(1 +(x[5]/K1)**q) - k2*x[1]*(N-x[4]) - k6*x[1]*x[4]
    xdot[2] = k2*x[1]*(N-x[4]) - k3*x[2]*(A-x[5])
    xdot[3] = k3*x[2]*(A-x[5]) - k4*x[3]*x[4] - kappa*(x[3]-x[6])
    xdot[4] = k2*x[1]*(N-x[4]) - k4*x[3]*x[4] - k6*x[1]*x[4]
    xdot[5] = -2*k1*x[0]*x[5]/(1 +(x[5]/K1)**q) + 2*k3*x[2]*(A-x[5]) - k5*x[5]
    xdot[6] = mu*kappa*(x[3]-x[6]) -k7*x[6]
    return xdot


def data_gen_sys_arc4s(sys_params, N_CURVES,SYSTEM_NO):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
        # get_input = input('Do you wanna delete the existing system[y/n]? ')
        # if get_input == 'y':
        #     shutil.rmtree(storage_folder)
        #     os.mkdir(storage_folder)
        # else:
        #     return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data = {}
    plt.figure()
    X0 = np.empty(shape=(0,4))
    t = np.arange(0, sys_params['t_end'], sys_params['Ts'])
    for i in range(N_CURVES):
        x0_curr = np.random.uniform(sys_params['x_min'],sys_params['x_max'],size=(4))
        X0 = np.concatenate([X0,x0_curr.reshape(1,-1)],axis=0)
        X = odeint(activator_repressor_clock_4states, x0_curr, t, args = sys_params['sys_params_arc4s'])
        Y = sys_params['k_3n'] * X[:, 1:2] / (sys_params['k_3d'] + X[:, 3:4])
        dict_indexed_data[i]= {'X':X,'Y':Y}
        plt.plot(dict_indexed_data[i]['X'][:,1],dict_indexed_data[i]['X'][:,3])
    plt.plot(X0[:,1],X0[:,3],'*')
    plt.show()
    plt.figure()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data,SYSTEM_NO)
    return

def data_gen_sys_arc2s(sys_params, N_CURVES,SYSTEM_NO):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
        # get_input = input('Do you wanna delete the existing system[y/n]? ')
        # if get_input == 'y':
        #     shutil.rmtree(storage_folder)
        #     os.mkdir(storage_folder)
        # else:
        #     return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data = {}
    plt.figure()
    X0 = np.empty(shape=(0,2))
    t = np.arange(0, sys_params['t_end'], sys_params['Ts'])
    for i in range(N_CURVES):
        x0_curr = np.random.uniform(sys_params['x_min'],sys_params['x_max'])
        X0 = np.concatenate([X0,x0_curr.reshape(1,-1)],axis=0)
        X = odeint(activator_repressor_clock_2states, x0_curr, t, args = sys_params['sys_params_arc2s'])
        Y = sys_params['k_3n'] * X[:, 1:2]**sys_params['cooperativity_1'] / (1 + sys_params['k_3d']*X[:, 0:1]**sys_params['cooperativity_2'])
        dict_indexed_data[i]= {'X':X,'Y':Y}
        plt.plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])
    plt.plot(X0[:,0],X0[:,1],'*')
    plt.show()
    plt.figure()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data,SYSTEM_NO)
    return

def data_gen_sys_duffing_oscillator(sys_params, N_CURVES,SYSTEM_NO):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
        # get_input = input('Do you wanna delete the existing system[y/n]? ')
        # if get_input == 'y':
        #     shutil.rmtree(storage_folder)
        #     os.mkdir(storage_folder)
        # else:
        #     return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data = {}
    plt.figure()
    X0 = np.empty(shape=(0,2))
    t = np.arange(0, sys_params['t_end'], sys_params['Ts'])
    for i in range(N_CURVES):
        x0_curr = np.random.uniform(sys_params['x_min'],sys_params['x_max'])
        X0 = np.concatenate([X0,x0_curr.reshape(1,-1)],axis=0)
        X = odeint(duffing_oscillator, x0_curr, t, args = sys_params['sys_params_duffosc'])
        print(X.shape)
        Y1 = np.exp(sys_params['gamma'] * X[:, 1:2])*np.cos(X[:, 0:1])
        Y2 = np.exp(X[:, 0:1] + X[:, 1:2])
        Y = np.concatenate([Y1,Y2],axis=1)
        print(Y.shape)
        dict_indexed_data[i]= {'X':X,'Y':Y}
        plt.plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])
    plt.plot(X0[:,0],X0[:,1],'*')
    plt.show()
    plt.figure()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data,SYSTEM_NO)
    return

def data_gen_sys_MEMS_accelerometer(sys_params, N_CURVES,SYSTEM_NO):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
        # get_input = input('Do you wanna delete the existing system[y/n]? ')
        # if get_input == 'y':
        #     shutil.rmtree(storage_folder)
        #     os.mkdir(storage_folder)
        # else:
        #     return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data = {}
    plt.figure()
    X0 = np.empty(shape=(0,2))
    t = np.arange(0, sys_params['t_end'], sys_params['Ts'])
    for i in range(N_CURVES):
        x0_curr = np.random.uniform(sys_params['x_min'],sys_params['x_max'])
        X0 = np.concatenate([X0,x0_curr.reshape(1,-1)],axis=0)
        X = odeint(MEMS_accelerometer, x0_curr, t, args = sys_params['sys_params_MEMS_accel'])
        print(X.shape)
        Y = -X[:, 0:1]*sys_params['Vs']/ (sys_params['d'] + X[:, 0:1])
        print(Y.shape)
        dict_indexed_data[i]= {'X':X,'Y':Y}
        plt.plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])
    plt.plot(X0[:,0],X0[:,1],'*')
    plt.show()
    plt.figure()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data,SYSTEM_NO)
    return

def data_gen_sys_combinatorial_promoter(sys_params, N_CURVES,SYSTEM_NO):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        get_input = input('Do you wanna delete the existing system[y/n]? ')
        if get_input == 'y':
            shutil.rmtree(storage_folder)
            os.mkdir(storage_folder)
        else:
            return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data = {}
    # plt.figure()
    X0 = np.empty(shape=(0,11))
    t = np.arange(0, sys_params['t_end'], sys_params['Ts'])
    for i in range(N_CURVES):
        x0_curr = np.array([np.random.uniform(sys_params['x_min'][i],sys_params['x_max'][i]) for i in range(len(sys_params['x_min']))])
        X0 = np.concatenate([X0,x0_curr.reshape(1,-1)],axis=0)
        X = odeint(combinatorial_promoter, x0_curr, t, args = sys_params['sys_params_arc4s'])
        Y = 1 / (1 + (sys_params['Ka'] / X[:, 10:11])**0.5)
        dict_indexed_data[i]= {'X':X,'Y':Y}
        # plt.plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])
    # plt.plot(X0[:,0],X0[:,1],'o',color='salmon',fillstyle='none',markersize=5)
    # plt.show()
    # plt.figure()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data, SYSTEM_NO)
    return

def data_gen_sys_glycolytic_oscillator(sys_params, N_CURVES,SYSTEM_NO,DOWNSAMPLE_FACTOR = -1):
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        get_input = input('Do you wanna delete the existing system[y/n]? ')
        if get_input == 'y':
            shutil.rmtree(storage_folder)
            os.mkdir(storage_folder)
        else:
            return
    else:
        os.mkdir(storage_folder)
    # Simulation
    dict_indexed_data_in = {}
    # plt.figure()
    X0 = np.empty(shape=(0,7))
    t = np.arange(0, sys_params['t_end'], sys_params['Ts'])
    for i in range(N_CURVES):
        x0_curr = np.array([np.random.uniform(sys_params['x_min'][i],sys_params['x_max'][i]) for i in range(len(sys_params['x_min']))])
        X0 = np.concatenate([X0,x0_curr.reshape(1,-1)],axis=0)
        X = odeint(glycolytic_oscillator, x0_curr, t, args = sys_params['sys_params_arc4s'])
        Y = 1 / (1 + (sys_params['Ka'] / X[:, 6:7])**0.5)
        dict_indexed_data_in[i]= {'X':X,'Y':Y}
        # plt.plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])

    if DOWNSAMPLE_FACTOR == -1:
        dict_indexed_data_out = dict_indexed_data_in
    else:
        dict_indexed_data_out = {}
        for curve in range(N_CURVES):
            dict_indexed_data_out[curve] = {}
            # Downsample the data in X and ignore the last sample
            dict_indexed_data_out[curve]['X'] = dict_indexed_data_in[curve]['X'][::DOWNSAMPLE_FACTOR][:-1]
            # Concatenate the data in Y between the samples as the output
            Y = np.empty(shape =(0, DOWNSAMPLE_FACTOR*dict_indexed_data_in[curve]['Y'].shape[1]))
            for i in range(dict_indexed_data_out[curve]['X'].shape[0]):
                Y = np.concatenate([Y,dict_indexed_data_in[curve]['Y'][i*DOWNSAMPLE_FACTOR:(i+1)*DOWNSAMPLE_FACTOR,:].reshape(1,-1)],axis=0)
            dict_indexed_data_out[curve]['Y'] = Y
    # plt.plot(X0[:,0],X0[:,1],'o',color='salmon',fillstyle='none',markersize=5)
    # plt.show()
    # plt.figure()
    sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data_out,SYSTEM_NO)
    return



# ----------------------------------------------------------------------------------------------------------------
# DATA POST PROCESSING
# ----------------------------------------------------------------------------------------------------------------

def model_prediction(dict_indexed_data, dict_params, SYSTEM_NUMBER):
    dict_indexed_data_predictions = {}
    for data_index in dict_indexed_data.keys():
        dict_DATA_i = scale_data_using_existing_scaler_folder(dict_indexed_data[data_index], SYSTEM_NUMBER)
        X_scaled = dict_DATA_i['X']
        Y_scaled = dict_DATA_i['Y']
        n_base_states = X_scaled.shape[1]
        psiX = dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: X_scaled})
        # One Step Prediction ----------------------
        psixpT_i = psiX[0,:]
        psiX_est_one_step = copy.deepcopy(psiX)
        for i in range(0, X_scaled.shape[0] - 1):
            psiX_est_one_step[i + 1, :] = np.matmul(psiX[i,:], dict_params['KxT_num'])
        Y_est_one_step = np.matmul(psiX_est_one_step, dict_params['WhT_num'])
        # N Step Prediction ----------------------
        psixpT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[0:1, :]})
        psiX_est_n_step = copy.deepcopy(psiX)
        for i in range(0, X_scaled.shape[0] - 1):
            psixfT_i = np.matmul(psixpT_i, dict_params['KxT_num'])
            psiX_est_n_step[i + 1, :] = psixfT_i
            psixpT_i = psixfT_i
        Y_est_n_step = np.matmul(psiX_est_n_step, dict_params['WhT_num'])
        dict_indexed_data_predictions[data_index] = {}
        dict_indexed_data_predictions[data_index]['X'] = dict_indexed_data[data_index]['X']
        dict_indexed_data_predictions[data_index]['X_est_one_step'] = inverse_transform_X(psiX_est_one_step[:, 0:n_base_states], SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['X_est_n_step'] = inverse_transform_X(psiX_est_n_step[:, 0:n_base_states], SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['X_scaled'] = X_scaled
        dict_indexed_data_predictions[data_index]['X_scaled_est_one_step'] = psiX_est_one_step[:, 0:n_base_states]
        dict_indexed_data_predictions[data_index]['X_scaled_est_n_step'] = psiX_est_n_step[:, 0:n_base_states]
        dict_indexed_data_predictions[data_index]['Y'] = dict_indexed_data[data_index]['Y']
        dict_indexed_data_predictions[data_index]['Y_est_one_step'] = inverse_transform_Y(Y_est_one_step, SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['Y_est_n_step'] = inverse_transform_Y(Y_est_n_step, SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['Y_scaled'] = Y_scaled
        dict_indexed_data_predictions[data_index]['Y_scaled_est_one_step'] = Y_est_one_step
        dict_indexed_data_predictions[data_index]['Y_scaled_est_n_step'] = Y_est_n_step
        dict_indexed_data_predictions[data_index]['psiX'] = psiX
        dict_indexed_data_predictions[data_index]['psiX_est_one_step'] = psiX_est_one_step
        dict_indexed_data_predictions[data_index]['psiX_est_n_step'] = psiX_est_n_step
    return dict_indexed_data_predictions

def model_prediction_state_only(dict_indexed_data, dict_params, SYSTEM_NUMBER):
    dict_indexed_data_predictions = {}
    for data_index in dict_indexed_data.keys():
        dict_DATA_i = scale_data_using_existing_scaler_folder(dict_indexed_data[data_index], SYSTEM_NUMBER)
        X_scaled = dict_DATA_i['X']
        n_base_states = X_scaled.shape[1]
        psiX = dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: X_scaled})
        # One Step Prediction ----------------------
        psixpT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[0:1, :]})
        psiX_est_one_step = copy.deepcopy(psiX)
        for i in range(0, X_scaled.shape[0] - 1):
            psixfT_i = np.matmul(psixpT_i, dict_params['KxT_num'])
            psiX_est_one_step[i + 1, :] = psixfT_i
            psixpT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[i + 1:i + 2, :]})
        # N Step Prediction ----------------------
        psixpT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: X_scaled[0:1, :]})
        psiX_est_n_step = copy.deepcopy(psiX)
        for i in range(0, X_scaled.shape[0] - 1):
            psixfT_i = np.matmul(psixpT_i, dict_params['KxT_num'])
            psiX_est_n_step[i + 1, :] = psixfT_i
            psixpT_i = psixfT_i
        dict_indexed_data_predictions[data_index] = {}
        dict_indexed_data_predictions[data_index]['X'] = dict_indexed_data[data_index]['X']
        dict_indexed_data_predictions[data_index]['X_est_one_step'] = inverse_transform_X(psiX_est_one_step[:, 0:n_base_states], SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['X_est_n_step'] = inverse_transform_X(psiX_est_n_step[:, 0:n_base_states], SYSTEM_NUMBER)
        dict_indexed_data_predictions[data_index]['X_scaled'] = X_scaled
        dict_indexed_data_predictions[data_index]['X_scaled_est_one_step'] = psiX_est_one_step[:, 0:n_base_states]
        dict_indexed_data_predictions[data_index]['X_scaled_est_n_step'] = psiX_est_n_step[:, 0:n_base_states]
        dict_indexed_data_predictions[data_index]['psiX'] = psiX
        dict_indexed_data_predictions[data_index]['psiX_est_one_step'] = psiX_est_one_step
        dict_indexed_data_predictions[data_index]['psiX_est_n_step'] = psiX_est_n_step
    return dict_indexed_data_predictions

def observables_and_eigenfunctions(dict_params,sampling_resolution):
    # psi --- observables
    # phi --- eigenfunctions
    x1 = np.arange(0, 1 + sampling_resolution, sampling_resolution)
    x2 = np.arange(0, 1 + sampling_resolution, sampling_resolution)
    X1, X2 = np.meshgrid(x1, x2)
    eigval, L_eigvec = np.linalg.eig(dict_params['KxT_num']) # These are the left eigenvectors of K
    # Eigenfunctions are given by psixT*L_eigvec
    # AT*W = W*LAMBDA ==> psixT[k+1] = psixT[k]*K = [psixT[k]*W] * LAMBDA *inv(W)
    # Extracting info about number of states by using a dummy transformation
    psiXpT_trial = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: np.zeros(shape=(1, 2))})
    n_LiftedStates = psiXpT_trial.shape[1]
    # Getting the observables and eigenfunctions
    PSI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_LiftedStates))
    PHI = np.zeros(shape=(X1.shape[0], X1.shape[1], n_LiftedStates))
    for i, j in itertools.product(range(X1.shape[0]), range(X1.shape[1])):
        x1_i = X1[i, j]
        x2_i = X2[i, j]
        psiXT_i = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: np.array([[x1_i, x2_i]])})
        PSI[i, j, :] = psiXT_i.reshape(-1)
        PHI[i, j, :] = np.matmul(psiXT_i,L_eigvec).reshape(-1)
    # PSI[:,:,i] gives the i-th observable
    # PHI[:,:,i] gives the i-th eigenfunction
    dict_out = {'X1':X1, 'X2':X2, 'observables': PSI, 'eigenfunctions':PHI}
    return dict_out
