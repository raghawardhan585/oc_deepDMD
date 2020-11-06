##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

SYSTEM_NO = 1


sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_trend_data = {}
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
        training_error = df_run_info.loc['training error', df_run_info.columns[-1]]
        validation_error = df_run_info.loc['validation error', df_run_info.columns[-1]]
        dict_trend_data[RUN_NO]={'n_nodes':n_nodes,'n_layers':n_layers,'n_observables':n_observables,'training_error':training_error,'validation_error':validation_error}
df_trend_data = pd.DataFrame(dict_trend_data).T.sort_index()

# n_layers is constant
N_LAYERS = 3
unique_n_observables = df_trend_data.n_observables.unique()
unique_n_nodes = df_trend_data.n_nodes.unique()

dict_train_error_mean = {}
dict_train_error_std = {}
dict_valid_error_mean = {}
dict_valid_error_std = {}
for N_obs in unique_n_observables:
    dict_train_error_mean[N_obs] = {}
    dict_train_error_std[N_obs] = {}
    dict_valid_error_mean[N_obs] = {}
    dict_valid_error_std[N_obs] = {}
    for N_nod in unique_n_nodes:
        df_n_obs_filter = df_trend_data[df_trend_data.n_observables.isin([N_obs])]
        df_n_nod_filter = df_n_obs_filter[df_n_obs_filter.n_nodes.isin([N_nod])]
        dict_train_error_mean[N_obs][N_nod] = df_n_nod_filter.loc[:,'training_error'].mean()
        dict_train_error_std[N_obs][N_nod] = df_n_nod_filter.loc[:, 'training_error'].std()
        dict_valid_error_mean[N_obs][N_nod] = df_n_nod_filter.loc[:, 'validation_error'].mean()
        dict_valid_error_std[N_obs][N_nod] = df_n_nod_filter.loc[:, 'training_error'].std()



plt.figure()
plt.plot




