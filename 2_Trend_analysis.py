##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os


colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette



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


df_train_error_mean = pd.DataFrame(dict_train_error_mean).sort_index().T.sort_index()
df_train_error_std = pd.DataFrame(dict_train_error_std).sort_index().T.sort_index()
df_valid_error_mean = pd.DataFrame(dict_valid_error_mean).sort_index().T.sort_index()
df_valid_error_std = pd.DataFrame(dict_valid_error_std).sort_index().T.sort_index()

leg_entries = []
for i in df_train_error_mean.columns:
    leg_entries.append(str(int(i)) + ' nodes')

plt.figure()
# plt.plot(df_train_error_mean.index,df_train_error_mean)
for i in df_train_error_mean.columns:
    plt.errorbar(df_train_error_mean.index,df_train_error_mean.loc[:,i], yerr =df_train_error_std.loc[:,i],uplims=True, lolims=True)

plt.title('Training Error')
plt.xlabel('Number of observables')
plt.xticks(df_train_error_mean.index,df_train_error_mean.index)
plt.ylabel('Error')
plt.legend(leg_entries)
plt.show()

plt.figure()
# plt.plot(df_valid_error_mean.index,df_valid_error_mean)
for i in df_valid_error_mean.columns:
    plt.errorbar(df_valid_error_mean.index,df_valid_error_mean.loc[:,i], yerr =df_valid_error_std.loc[:,i],uplims=True, lolims=True)
plt.title('Validation Error')
plt.xlabel('Number of observables')
plt.xticks(df_valid_error_mean.index,df_valid_error_mean.index)
plt.ylabel('Error')
plt.legend(leg_entries)
plt.show()




