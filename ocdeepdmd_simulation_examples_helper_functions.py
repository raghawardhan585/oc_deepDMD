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

def scale_train_data(dict_DATA_IN,method ='standard'):
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
        X_Scale = StandardScaler().fit(X_all)
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
            Y_Scale= StandardScaler().fit(Y_all)
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


def generate_2state_initial_condition():
    r = np.random.uniform(5.,10.)
    theta = np.random.uniform(0.,2*np.pi)
    return np.array([[r*np.cos(theta),r*np.sin(theta)]])

def sim_sys_1_2(sys_params):
    if sys_params['x0'].shape != (1,2):
        print('[ERROR]: Incorrect dimensions of the initial condition x0. It should be array of (1,2)')
        exit(1)
    X = sys_params['x0']
    for i in range(sys_params['N_data_points']-1):
        # Autonomous Linear Components0
        x1_next = sys_params['A'][0,0] * X[-1, 0] + sys_params['A'][0,1] * X[-1, 0] #+ sys_params['delta']* U[i,0]**2
        x2_next = sys_params['A'][1,0] * X[-1, 1] + sys_params['A'][1,1] * X[-1,0] #+ sys_params['epsilon']*X[-1,1]**2*U[i,0]
        # Nonlinear dynamics components
        x2_next = x2_next + sys_params['gamma'] * X[-1,0]**2
        X = np.concatenate([X,np.array([[x1_next,x2_next]])],axis=0)
    Y = X[:,0:1] * X[:,1:2]
    dict_DATA = {'X': X, 'Y': Y}
    return dict_DATA

def data_gen_sys_1_2(sys_params, N_CURVES,SYSTEM_NO):
    # Simulation
    dict_indexed_data = {}
    plt.figure()
    X0 = np.empty(shape=(0,2))
    for i in range(N_CURVES):
        r = np.random.uniform(5., 10.)
        theta = np.random.uniform(0., 2 * np.pi)
        sys_params['x0'] = generate_2state_initial_condition()
        X0 = np.concatenate([X0,sys_params['x0']],axis=0)
        dict_indexed_data[i]= sim_sys_1_2(sys_params)
        plt.plot(dict_indexed_data[i]['X'][:,0],dict_indexed_data[i]['X'][:,1])
    plt.plot(X0[:,0],X0[:,1],'*')
    # Sorting into deep DMD format
    Xp = np.empty((0,2))
    Xf = np.empty((0,2))
    Yp = np.empty((0,1))
    Yf = np.empty((0,1))
    ls_all_indices = np.arange(N_CURVES)
    random.shuffle(ls_all_indices)
    for i in ls_all_indices:
        Xp = np.concatenate([Xp,dict_indexed_data[i]['X'][0:-1,:]],axis=0)
        Xf = np.concatenate([Xf,dict_indexed_data[i]['X'][1:,:]],axis=0)
        Yp = np.concatenate([Yp,dict_indexed_data[i]['Y'][0:-1,:]],axis=0)
        Yf = np.concatenate([Yf,dict_indexed_data[i]['Y'][1:,:]],axis=0)
    dict_DATA_RAW = {'Xp': Xp, 'Xf': Xf, 'Yp': Yp, 'Yf': Yf}
    dict_DATA,dict_Scaler,_ = scale_train_data(dict_DATA_RAW,'min max')
    # Create a folder for the system and store the data
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        get_input = input('Do you wanna delete the existing system[y/n]? ')
        if get_input == 'y':
            shutil.rmtree(storage_folder)
            os.mkdir(storage_folder)
            with open(storage_folder + '/System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
                pickle.dump(dict_DATA,handle)
            with open(storage_folder + '/System_'+ str(SYSTEM_NO) + '_SimulatedData.pickle', 'wb') as handle:
                pickle.dump(dict_indexed_data,handle)
            with open(storage_folder + '/System_'+ str(SYSTEM_NO) +'_OrderedIndices.pickle', 'wb') as handle:
                pickle.dump(ls_all_indices,handle)
            with open(storage_folder + '/System_'+ str(SYSTEM_NO) +'_DataScaler.pickle', 'wb') as handle:
                pickle.dump(dict_Scaler,handle)
            # Store the data in Koopman
            with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_'+ str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
                pickle.dump(dict_DATA,handle)
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
            RUN_NO = 0
            for i in dict_run_conditions.keys():
                if i in [0, 1, 2, 3]:
                    for j in range(NO_OF_ITERATIONS_PER_GPU):
                        general_run = 'python3 gen_control_Koopman_SHARA_addition.py \'/gpu:' + str(i) + '\' ' + str(
                            SYSTEM_NO) + ' 0 1 0 '
                        run_params = str(RUN_NO)
                        for items in dict_run_conditions[i].keys():
                            run_params = run_params + ' ' + str(dict_run_conditions[i][items])
                        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(RUN_NO) + '.txt &\n'
                        bash_exec.write(general_run + run_params + write_to_file)
                        RUN_NO = RUN_NO + 1
                elif i == 4:
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
    return

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
    destination_folder = 'System_' + str(SYSTEM_NUMBER)
    current_run_no = -1
    for items in os.listdir(destination_folder):
        if items[0:4] == 'RUN_':
            current_run_no = np.max([current_run_no, items[4:]])
    current_run_no = current_run_no + 1
    # Transfer the files
    for items in list(set(os.listdir(source_folder)) - {'.DS_Store'}):
        shutil.move(source_folder + '/' + items, destination_folder + '/RUN_' + str(current_run_no))
        shutil.move(runinfo_folder + '/' + items + '.txt',
                    destination_folder + '/RUN_' + str(current_run_no) + '/RUN_' + str(current_run_no) + '.txt')
        current_run_no = current_run_no + 1
    return

def get_all_run_info(SYSTEM_NO,RUN_NO,sess):
    sys_folder_name = 'System_' + str(SYSTEM_NO)
    run_folder_name = sys_folder_name + '/RUN_' + str(RUN_NO)
    with open(sys_folder_name+'/'+sys_folder_name+'_DataScaler.pickle','rb') as handle:
        data_Scaler = pickle.load(handle)
    with open(sys_folder_name+'/'+sys_folder_name+'_OrderedIndices.pickle','rb') as handle:
        ls_all_indices = pickle.load(handle)
    with open(sys_folder_name+'/'+sys_folder_name+'_SimulatedData.pickle','rb') as handle:
        dict_indexed_data = pickle.load(handle) # No scaling appplied here
    with open(sys_folder_name+'/'+sys_folder_name+'_ocDeepDMDdata.pickle','rb') as handle:
        dict_DATA = pickle.load(handle) # Scaled data
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
    return dict_params, data_Scaler, ls_all_indices, dict_indexed_data, dict_DATA, df_train_learning_curves,df_run_info