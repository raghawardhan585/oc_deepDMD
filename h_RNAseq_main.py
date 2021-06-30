##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
from scipy.signal import savgol_filter as sf
from sklearn.preprocessing import MinMaxScaler
import pickle
import random
import numpy as np
import pandas as pd
import os
import shutil
import random
import matplotlib.pyplot as plt
import re
import copy
import itertools
import seaborn as sb
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= True,n_outputs= -1) # Getting the raw RNAseq and OD600 data to the state and output data format
with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)


## Filter the data based on the specified genes

def get_data_with_required_genes(dict_data_in,ls_genes):
    ls_conditions = list(dict_data_in.keys())
    ls_indices = list(dict_data_in[ls_conditions[0]].keys())
    dict_data_out = copy.deepcopy(dict_data_in)
    for cond,i in itertools.product(ls_conditions,ls_indices):
        dict_data_out[cond][i]['df_X_TPM'] = dict_data_out[cond][i]['df_X_TPM'].loc[ls_genes,:]
    return dict_data_out

def get_data_with_averaged_MIN_NC_timepoints_1_2(dict_data_in):
    dict_data_out = copy.deepcopy(dict_data_in)
    for cond,curve in itertools.product(['MN','NC'],dict_data_in['MN']):
        dict_data_out[cond][curve]['df_X_TPM'].insert(0, 2, list(dict_data_out[cond][curve]['df_X_TPM'].loc[:, 0:3].mean(axis=1)), True)
        dict_data_out[cond][curve]['df_X_TPM'].insert(0, 1, list(dict_data_out[cond][curve]['df_X_TPM'].loc[:, 0:3].mean(axis=1)), True)
    return dict_data_out

def get_data_savgol_filter(dict_data_in, n_data_pts = 5, polyorder = 3):
    ls_conditions = list(dict_data_in.keys())
    ls_indices = list(dict_data_in[ls_conditions[0]].keys())
    dict_data_out = copy.deepcopy(dict_data_in)
    for cond, i in itertools.product(ls_conditions, ls_indices):
        dict_data_out[cond][i]['df_X_TPM'] = pd.DataFrame(sf(dict_data_out[cond][i]['df_X_TPM'],n_data_pts,polyorder,axis=1),columns= dict_data_out[cond][i]['df_X_TPM'].columns,index=dict_data_out[cond][i]['df_X_TPM'].index)
    return dict_data_out


def formulate_and_save_MAX_Koopman_Data_h(dict_data,SYSTEM_NO=0):
    ls_all_indices = list(dict_data['MX'].keys())
    random.shuffle(ls_all_indices)
    dict_indices = {'train':ls_all_indices[0:12],'valid': ls_all_indices[12:14], 'test': ls_all_indices[14:16]}
    n_states = dict_data['MX'][0]['df_X_TPM'].shape[0]
    n_outputs = dict_data['MX'][0]['Y'].shape[0]
    # dict_data.keys() = ['scaled' - [0,1,...16], 'unscaled' - [0,1,...16], 'index' - ['train','valid','test'], 'X_scaler', 'Y_scaler']
    X_train = np.empty(shape=(0,n_states))
    Y_train = np.empty(shape=(0,n_outputs))
    Y_train2 = np.empty(shape=(0, 1))
    for i in dict_indices['train']:
        X_train = np.concatenate([X_train,np.array(dict_data['MX'][i]['df_X_TPM']).T],axis=0)
        Y_train = np.concatenate([Y_train, np.array(dict_data['MX'][i]['Y']).T], axis=0)
        Y_train2 = np.concatenate([Y_train2, np.array(dict_data['MX'][i]['Y']).reshape(-1,1)], axis=0)
    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)
    Y_scaler = MinMaxScaler()
    Y_scaler.fit(Y_train)
    Y_scaler2 = MinMaxScaler()
    Y_scaler2.fit(Y_train2)
    dict_DMD_data = {'index': dict_indices, 'X_scaler': X_scaler,'Y_scaler': Y_scaler, 'unscaled': {}, 'scaled': {}, 'original': {}}
    for i in ls_all_indices:
        dict_DMD_data['original'][i] = copy.deepcopy(dict_data['MX'][i])
        dict_DMD_data['unscaled'][i] = {'XT': np.array(dict_data['MX'][i]['df_X_TPM']).T, 'YT':np.array(dict_data['MX'][i]['Y']).T}
        # dict_DMD_data['scaled'][i] = {'XT': X_scaler.transform(dict_DMD_data['unscaled'][i]['XT']), 'YT': Y_scaler.transform(dict_DMD_data['unscaled'][i]['YT']), 'YT2': Y_scaler2.transform(dict_DMD_data['unscaled'][i]['YT'].reshape(-1,1))}
        dict_DMD_data['scaled'][i] = {'XT': X_scaler.transform(dict_DMD_data['unscaled'][i]['XT']),
                                      'YT': Y_scaler.transform(dict_DMD_data['unscaled'][i]['YT'])}
    storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/h_OCdeepDMD' + '/System_' + str(SYSTEM_NO)
    if os.path.exists(storage_folder):
        get_input = input('Do you wanna delete the existing system[y/n]? ')
        # get_input = 'y'
        if get_input == 'y':
            shutil.rmtree(storage_folder)
            os.mkdir(storage_folder)
        else:
            quit(0)
    else:
        os.mkdir(storage_folder)
    # Store the data in Box folder
    with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
        pickle.dump(dict_DMD_data, handle)
    # Store the data in Koopman
    with open('/Users/shara/Desktop/oc_deepDMD/h_OCdeepDMD_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
        pickle.dump(dict_DMD_data, handle)
    return


##
# ls_conditions = ['MX']
# ls_genes_choice = ['PP_1223','PP_1342']
SYSTEM_NO = 0
dict_data1 = copy.deepcopy(dict_DATA_ORIGINAL)
dict_data1 = get_data_with_averaged_MIN_NC_timepoints_1_2(dict_data1)
dict_data1_filtered1 = get_data_with_required_genes(dict_data1,ls_genes = ['PP_1223','PP_1342'])
dict_data1_filtered1 = get_data_savgol_filter(dict_data1_filtered1)
formulate_and_save_MAX_Koopman_Data_h(dict_data1_filtered1, SYSTEM_NO=SYSTEM_NO)


data_directory = 'h_OCdeepDMD_data/'
data_suffix = 'System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
file_path = data_directory + data_suffix
with open(file_path, 'rb') as handle:
    dict_data = pickle.load(handle)

##
ls_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure()
curve = 0
f,ax = plt.subplots(2,1)
ax[0].plot(np.arange(1,8,0.05),dict_data['unscaled'][curve]['YT'].reshape(-1),'.',color = ls_colors[0])
ax[0].plot(np.arange(1,8,1),dict_data['unscaled'][curve]['YT'],'.',color = ls_colors[1])

ax[1].plot(np.arange(1,8,0.05),dict_data['scaled'][curve]['YT2'],'.',color = ls_colors[3])
ax[1].plot(np.arange(1,8,1),dict_data['scaled'][curve]['YT2'].reshape((7,-1)),'.',color = ls_colors[4])
ax[1].plot(np.arange(1,8,1),dict_data['scaled'][curve]['YT'],'.',color = ls_colors[2])
plt.show()

##






