##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
import pickle
import random
import numpy as np
import pandas as pd
import os
import shutil
import random
import matplotlib.pyplot as plt
import copy
import itertools
import seaborn as sb
import tensorflow as tf

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.metrics import make_scorer,r2_score
pd.set_option("display.max_rows", None, "display.max_columns", None)
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

# def ADD_BIAS_ROW(X_IN,ADD_BIAS):
#     if ADD_BIAS:
#         X_OUT = np.concatenate([X_IN, np.ones(shape=(1, X_IN.shape[1]))], axis=0)
#     else:
#         X_OUT = X_IN
#     return X_OUT
# def ADD_BIAS_COLUMN(X_IN,ADD_BIAS):
#     if ADD_BIAS:
#         X_OUT = np.concatenate([X_IN, np.ones(shape=(X_IN.shape[0], 1))], axis=1)
#     else:
#         X_OUT = X_IN
#     return X_OUT
# def REMOVE_BIAS_ROW(X_IN,ADD_BIAS):
#     if ADD_BIAS:
#         X_OUT = X_IN[0:-1,:]
#     else:
#         X_OUT = X_IN
#     return X_OUT
# def REMOVE_BIAS_COLUMN(X_IN,ADD_BIAS):
#     if ADD_BIAS:
#         X_OUT = X_IN[:,0:-1]
#     else:
#         X_OUT = X_IN
#     return X_OUT
# def resolve_complex_right_eigenvalues(E, W):
#     eval = copy.deepcopy(np.diag(E))
#     comp_modes = []
#     comp_modes_conj = []
#     for i1 in range(E.shape[0]):
#         if np.imag(E[i1, i1]) != 0:
#             print(i1)
#             # Find the complex conjugate
#             for i2 in range(i1 + 1, E.shape[0]):
#                 if eval[i2] == eval[i1].conj():
#                     break
#             # i1 and i2 are the indices of the complex conjugate eigenvalues
#             comp_modes.append(i1)
#             comp_modes_conj.append(i2)
#             E[i1, i1] = np.real(eval[i1])
#             E[i2, i2] = np.real(eval[i1])
#             E[i1, i2] = np.imag(eval[i1])
#             E[i2, i1] = - np.imag(eval[i1])
#             u1 = copy.deepcopy(np.real(W[:, i1:i1 + 1]))
#             w1 = copy.deepcopy(np.imag(W[:, i1:i1 + 1]))
#             W[:, i1:i1 + 1] = u1
#             W[:, i2:i2 + 1] = w1
#     E_out = np.real(E)
#     W_out = np.real(W)
#     return E_out, W_out, comp_modes, comp_modes_conj
#
# # ## Importing all necessary information
# SYSTEM_NO = 400
# ALL_CONDITIONS = ['MX','MN']
# ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
# # original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
# # indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
# # # Import the training data
# with open(ocdeepDMD_data_path,'rb') as handle:
#     dict_DMD_train_unbiased = pickle.load(handle) # already scaled, it is in transposed format though not stated
#
# # Xp_train, Xp_valid, Xf_train, Xf_valid =  train_test_split(dict_DMD_train_unbiased['Xp'],dict_DMD_train_unbiased['Xf'],train_size=0.875, shuffle=False)
# TRAIN_PERCENT = 87.5
#
# Xp = dict_DMD_train_unbiased['Xp']
# Xf = dict_DMD_train_unbiased['Xf']
# num_all_samples = len(Xp)
# num_trains = np.int(num_all_samples * TRAIN_PERCENT / 100)
# train_indices = np.arange(0, num_trains, 1)
# valid_indices = np.arange(num_trains,num_all_samples,1)
# dict_train = {}
# dict_valid = {}
# Xp_train = Xp[train_indices]
# Xp_valid = Xp[valid_indices]
# # dict_train['Yp'] = Yp[train_indices]
# # dict_valid['Yp'] = Yp[valid_indices]
# Xf_train = Xf[train_indices]
# Xf_valid = Xf[valid_indices]
# # dict_train['Yf'] = Yf[train_indices]
# # dict_valid['Yf'] = Yf[valid_indices]
#
# # Since the data is scaled, we use the affine transformation theorem to impose a bias constraint on the model
# dict_DMD_train = copy.deepcopy(dict_DMD_train_unbiased)
# dict_DMD_train['Xp'] = ADD_BIAS_COLUMN(dict_DMD_train['Xp'],True)
# dict_DMD_train['Xf'] = ADD_BIAS_COLUMN(dict_DMD_train['Xf'],True)
# # Import the full dataset and the indices
# with open(original_data_path,'rb') as handle:
#     dict_data_original = pickle.load(handle)
# # Import the ordered indices - Last two indices are used as testing data
# with open(indices_path,'rb') as handle:
#     ls_data_indices = pickle.load(handle)
# # ALL_CONDITIONS = list(dict_data_original.keys())
# # Setup the prediction dictionary for all datasets
# dict_DMD_results = {}
# for COND in ALL_CONDITIONS:
#     dict_DMD_results[COND] = {}
# for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
#     dict_DMD_results[COND][i] = {}
#     dict_DMD_results[COND][i]['Xp'] = np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:,0:-1])
#     dict_DMD_results[COND][i]['Xf'] = np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:])
#     dict_DMD_results[COND][i]['Yp'] = np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1])
#     dict_DMD_results[COND][i]['Yf'] = np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:])
# # The scalers can be called directly using functions
#
#
# ## How does the linear model fit on all the data
#
# lin_model1_X = LinearRegression(fit_intercept=True).fit(dict_DMD_train_unbiased['Xp'],dict_DMD_train_unbiased['Xf'])
# lin_model1_Y = LinearRegression(fit_intercept=True).fit(np.concatenate([dict_DMD_train_unbiased['Xf'],dict_DMD_train_unbiased['Xp']],axis=0),np.concatenate([dict_DMD_train_unbiased['Yf'],dict_DMD_train_unbiased['Yp']],axis=0))
# ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
# plt.figure()
# dict_start_time_Yf = {'MX':2, 'MN':4, 'NC':4}
# DOWNSAMPLE_FACTOR = 4
# ls_train_indices = list(dict_DMD_results[ALL_CONDITIONS[0]].keys())[0:-2]
# ls_test_indices = list(dict_DMD_results[ALL_CONDITIONS[0]].keys())[-2:]
# for COND_NO in range(len(ALL_CONDITIONS)):
#     COND = ALL_CONDITIONS[COND_NO]
#     x_time = np.ones(shape=(np.product(dict_DMD_results[COND][0]['Yf'].shape)))
#     x_time[0] = 0
#     x_time = dict_start_time_Yf[COND] + np.cumsum(np.ones(shape=(np.product(dict_DMD_results[COND][i]['Yf'].shape))) * 3 / 60)
#     for items in dict_DMD_results[COND].keys():
#         # Scale the Xp
#         data_scaled = oc.scale_data_using_existing_scaler_folder({'Xp': dict_DMD_results[COND][items]['Xp'].T, 'Xf': dict_DMD_results[COND][items]['Xf'].T, 'Yp': dict_DMD_results[COND][items]['Yp'].T, 'Yf': dict_DMD_results[COND][items]['Yf'].T}, SYSTEM_NO)
#         XpTs = data_scaled['Xp']
#         XfTs = data_scaled['Xf']
#         YpTs = data_scaled['Yp']
#         YfTs = data_scaled['Yf']
#
#         # Predict the Xf - 1 step
#         XfTs_hat = lin_model1_X.predict(XpTs)
#         YpTs_hat = lin_model1_Y.predict(XpTs)
#         YfTs_hat = lin_model1_Y.predict(XfTs_hat)
#
#
#         # Predict the Xf - n step
#         XfTsn_hat = XpTs[0:1,:]
#         for i in range(len(XfTs_hat)):
#             XfTsn_hat = np.concatenate([XfTsn_hat,lin_model1_X.predict(XfTsn_hat[-1:])],axis=0)
#         XfTsn_hat = XfTsn_hat[1:]
#         YfTsn_hat = lin_model1_Y.predict(XfTsn_hat)
#
#         # Reverse the Xfs
#         Xf_hat = oc.inverse_transform_X(XfTs_hat, SYSTEM_NO).T
#         Yp_hat = oc.inverse_transform_Y(YpTs_hat, SYSTEM_NO).T
#         Yf_hat = oc.inverse_transform_Y(YfTs_hat, SYSTEM_NO).T
#
#         Xfn_hat = oc.inverse_transform_X(XfTsn_hat, SYSTEM_NO).T
#         Yfn_hat = oc.inverse_transform_Y(YfTsn_hat, SYSTEM_NO).T
#
#         # if items in ls_train_indices:
#         #     plt.plot(x_time[0:-1:DOWNSAMPLE_FACTOR],
#         #              dict_DMD_results[COND][items]['Yf'].T.reshape(-1)[0:-1:DOWNSAMPLE_FACTOR], '.',
#         #              color=ls_colors[COND_NO])
#         #     plt.plot(x_time, Yfn_hat.T.reshape(-1), color = ls_colors[COND_NO])
#         if items in ls_test_indices:
#             plt.plot(x_time[0:-1:DOWNSAMPLE_FACTOR],
#                      dict_DMD_results[COND][items]['Yf'].T.reshape(-1)[0:-1:DOWNSAMPLE_FACTOR], '.',
#                      color=ls_colors[COND_NO])
#             if items == ls_test_indices[-1]:
#                 plt.plot(x_time, Yfn_hat.T.reshape(-1), color=ls_colors[COND_NO], label = COND)
#             else:
#                 plt.plot(x_time, Yfn_hat.T.reshape(-1), color = ls_colors[COND_NO])
#         # Compute and the r2 score
#         dict_DMD_results[COND][items]['r2_Xfs_1step'] = r2_score(XfTs, XfTs_hat,multioutput ='variance_weighted')
#         dict_DMD_results[COND][items]['r2_Yps_1step'] = r2_score(YpTs.reshape(-1), YpTs_hat.reshape(-1))
#         dict_DMD_results[COND][items]['r2_Yfs_1step'] = r2_score(YfTs.reshape(-1), YfTs_hat.reshape(-1))
#         dict_DMD_results[COND][items]['r2_Xfs_nstep'] = r2_score(XfTs, XfTsn_hat,multioutput ='variance_weighted')
#         dict_DMD_results[COND][items]['r2_Yfs_nstep'] = r2_score(YfTs.reshape(-1), YfTsn_hat.reshape(-1))
#
#         dict_DMD_results[COND][items]['r2_Xf_1step'] = r2_score(dict_DMD_results[COND][items]['Xf'].T, Xf_hat.T,multioutput ='variance_weighted')
#         dict_DMD_results[COND][items]['r2_Yp_1step'] = r2_score(dict_DMD_results[COND][items]['Yp'].reshape(-1), Yp_hat.reshape(-1))
#         dict_DMD_results[COND][items]['r2_Yf_1step'] = r2_score(dict_DMD_results[COND][items]['Yf'].reshape(-1), Yf_hat.reshape(-1))
#         dict_DMD_results[COND][items]['r2_Xf_nstep'] = r2_score(dict_DMD_results[COND][items]['Xf'].T, Xfn_hat.T,multioutput ='variance_weighted')
#         dict_DMD_results[COND][items]['r2_Yf_nstep'] = r2_score(dict_DMD_results[COND][items]['Yf'].reshape(-1), Yfn_hat.reshape(-1))
# # a0 = plt.plot([0, 2],color = ls_colors[0],linestyle ='solid',label='MAX',linewidth = 1)
# # a1 = plt.plot([0,2],color = ls_colors[1],linestyle ='solid',label = 'MIN',linewidth = 1)
# # a2 = plt.plot([0,2],color = ls_colors[2],linestyle ='solid',label = 'NC',linewidth = 1)
# # l2 = plt.legend((a0,a1,a2),('MAX','MIN','NC'),loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
# plt.legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
# plt.xlabel('Time [hrs]')
# plt.ylabel('Fitness Score')
# # plt.title('Testing Data')
# plt.show()
# df_results ={}
# for COND in ALL_CONDITIONS:
#     df_results[COND] = pd.DataFrame(dict_DMD_results[COND]).T.loc[:,['r2_Xfs_1step','r2_Yps_1step','r2_Yfs_1step','r2_Yp_1step','r2_Yf_1step','r2_Xfs_nstep','r2_Yfs_nstep','r2_Xf_1step','r2_Xf_nstep','r2_Yf_nstep']]
#     print(df_results[COND])
#
# for COND in ALL_CONDITIONS:
#     print('-------------------------')
#     print(COND, ' CONDITION TEST DATA')
#     print('-------------------------')
#     # print(df_results[COND].iloc[-2:, :])
#     print(df_results[COND].iloc[-2:, :].mean(axis=0))
# # ##
# # with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','rb') as handle:
# #     df_stats1 = pickle.load(handle)
# # df_stats_filt = np.maximum(df_stats1, 0)
# #
# # for i in df_stats_filt.columns:
# #     if i>0.5:
# #         df_stats_filt = df_stats_filt.drop(columns=[i])
# # for i in df_stats_filt.index:
# #     if np.sum(df_stats_filt.loc[i,:])==0:
# #         df_stats_filt = df_stats_filt.drop(index=[i])
# # plt.plot(df_stats_filt.columns, df_stats_filt.T, '.',color= '#ff7f0e')
# # # plt.plot(df_stats_filt.columns,df_stats_filt.median(axis=0),color = '#ff7f0e')
# # plt.plot(df_stats_filt.columns,df_stats_filt.mean(axis=0), color = '#1f77b4',marker='.',markersize = 10,alpha = 1)
# # plt.errorbar(df_stats_filt.columns,df_stats_filt.mean(axis=0),df_stats_filt.std(axis=0), color = '#1f77b4', capsize = 8,fmt='.',alpha = 0.8)
# # plt.xlabel('Lasso hyperparameter ($\lambda$)')
# # plt.ylabel('$r^2$')
# # plt.yticks([0,0.3,0.6,0.9])
# # plt.xticks([0,0.1,0.2,0.3,0.4])
# # plt.xlim([-0.005,0.4])
# # plt.ylim([-0.005,1])
# # plt.show()
#
#
#
#
#
# ## Modes of the system
# num_modes = 195
#
# # Construct the K matrix
# K1_without_intercept = np.concatenate([np.array(lin_model1_X.coef_),np.zeros(shape=(1,len(lin_model1_X.coef_)))],axis=0)
# K1_intercept = np.concatenate([lin_model1_X.intercept_.reshape(-1,1),np.array([[1]])],axis=0)
# K1 = np.concatenate([K1_without_intercept,K1_intercept], axis=1)
#
#
# # TODO Do an eigen decomposition and modal reduction
# #  (keeping in mind that complex eigenvalues might be in the system)
# E_complex,W_complex = np.linalg.eig(K1)
# E, W, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(np.diag(E_complex),W_complex)
# Winv = np.linalg.inv(W)
#
# # Plot of the eigenvalues
# plt.plot(np.abs(E_complex))
# plt.xlabel('Index of $\lambda$')
# plt.ylabel('$\lambda$')
# # plt.xlim([0,250])
# plt.show()
# plt.plot((np.cumsum(np.abs(E_complex)**2)/np.sum(np.abs(E_complex)**2)))
# plt.xlabel('Number of $\lambda$s ($n$)')
# plt.ylabel('$(\sum_{i=1}^n|\lambda_i|^2)/(\sum_{i=1}^N|\lambda_i|^2)$')
# plt.xlim([0,250])
# plt.show()
#
#
# ## TODO Plot of the evolution of the eigenfunctions
# n_funcs = 10
# f,ax = plt.subplots(n_funcs,1,sharex=True,figsize=(7,n_funcs*1.5))
# dict_x_index ={'MX': np.array([2,3,4,5,6,7]),'MN': np.array([4,5,6,7]),'NC': np.array([4,5,6,7])}
# for COND_NO in range(len(ALL_CONDITIONS)):
#     COND = ALL_CONDITIONS[COND_NO]
#     Xps = oc.scale_data_using_existing_scaler_folder({'Xp':dict_DMD_results[COND][0]['Xp'].T},SYSTEM_NO)['Xp'].T
#     Xps = np.concatenate([Xps, np.ones(shape=(1,len(Xps[0])))], axis = 0)
#     Phis = np.matmul(Winv, Xps)
#     Phis = Phis[0:-1,:]
#     Phi = oc.inverse_transform_X(Phis.T,SYSTEM_NO).T
#     for i in range(n_funcs):
#         if i==0:
#             ax[i].plot(dict_x_index[COND], Phi[i, :],label = COND)
#         else:
#             ax[i].plot(dict_x_index[COND], Phi[i,:])
#         ax[i].set_title('$\lambda = $'+ str(E[i][i]))
#
# ax[0].legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
# ax[-1].set_xlabel('Time [hrs]')
# f.show()
# ## TODO Plot of the modes reconstructing the profile of the dynamics
# W_unscaled = oc.inverse_transform_X(W[0:-1].T, SYSTEM_NO).T
# n_plots = 10
# plt.figure()
# plt.plot(W_unscaled[:,0:1])
# plt.show()
#
#
# #


## OC deepDMD runs


# Preprocessing files
SYSTEM_NO = 401
ALL_CONDITIONS = ['MX']
# ALL_CONDITIONS = ['MX','MN']#list(dict_data_original.keys())
ls_runs1 = list(range(0,60)) # SYSTEM 401
# ls_runs1 = list(range(64,90)) # SYSTEM 304
# ls_runs1 = list(range(0,60)) # SYSTEM 304

# ls_runs1 = list(range(4,60)) # SYSTEM 304
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'

# Indices [train, validation and test]
with open(indices_path,'rb') as handle:
    ls_data_indices = pickle.load(handle)
ls_train_indices = ls_data_indices[0:12]
ls_valid_indices = ls_data_indices[12:14]
ls_test_indices = ls_data_indices[14:16]
# Datasets [sorted as scaled and unscaled] and Conditions
with open(original_data_path,'rb') as handle:
    dict_data_original = pickle.load(handle)

Xp1=[]
Xf1=[]
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices[0:14]):
    try:
        Xp1 = np.concatenate([Xp1, np.array(dict_data_original[COND][i]['df_X_TPM'])[:,0:-1].T],axis=0)
        Xf1 = np.concatenate([Xf1, np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T], axis=0)
    except:
        Xp1 = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 0:-1].T
        Xf1 = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T
    print('cond ',COND,' i: ', i, ' shape:', len(Xp1))

n_genes = len(dict_data_original[ALL_CONDITIONS[0]][ls_data_indices[0]]['df_X_TPM'])

dict_empty_all_conditions ={}
for COND in ALL_CONDITIONS:
    dict_empty_all_conditions[COND] = {}

dict_scaled_data = copy.deepcopy(dict_empty_all_conditions)
dict_unscaled_data = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    dict_intermediate = oc.scale_data_using_existing_scaler_folder(
        {'Xp': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
         'Xf': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
         'Yp': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
         'Yf': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}, SYSTEM_NO)
    dict_scaled_data[COND][i] = {'XpT': dict_intermediate['Xp'], 'XfT': dict_intermediate['Xf'],
                                 'YpT': dict_intermediate['Yp'], 'YfT': dict_intermediate['Yf']}
    dict_unscaled_data[COND][i] = {'XpT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                   'XfT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
                                   'YpT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
                                   'YfT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}


# TODO - generate predictions for each curve and write down the error statistics for each run


ls_all_run_indices = []
for folder in os.listdir(root_run_file + '/Sequential'):
    if folder[0:4] == 'RUN_':  # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
ls_runs1 = set(ls_runs1).intersection(set(ls_all_run_indices))
# TODO - Open the predictions folder or create one if it doesn't exist
try:
    with open(dict_predict_STATS_file, 'rb') as handle:
        dict_predict_STATS = pickle.load(handle)
except:
    dict_predict_STATS = {}
# Generate the predictions for each run
for run in ls_runs1:
    print('RUN: ', run)
    sess = tf.InteractiveSession()
    run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    dict_params['psixpT'] = tf.get_collection('psixpT')[0]
    dict_params['psixfT'] = tf.get_collection('psixfT')[0]
    dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
    dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
    dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
    try:
        dict_params['ypT_feed'] = tf.get_collection('ypT_feed')[0]
        dict_params['yfT_feed'] = tf.get_collection('yfT_feed')[0]
        dict_params['WhT_num'] = sess.run(tf.get_collection('WhT')[0])
        with_output = True
    except:
        with_output = False
    # print('Run :', run, '  r2_train :', r2_score(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: Xf_train}), np.matmul(dict_params['psixpT'].eval(feed_dict ={dict_params['xpT_feed']: Xp_train}),dict_params['KxT_num']), multioutput='variance_weighted'))
    # print('         r2_valid :',
    #       r2_score(dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: Xf_valid}),
    #                np.matmul(dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: Xp_valid}),
    #                          dict_params['KxT_num']), multioutput='variance_weighted'))
    dict_instant_run_result = copy.deepcopy(dict_empty_all_conditions)
    for items in dict_instant_run_result.keys():
        dict_instant_run_result[items] = {'train_Xf_1step': [], 'train_Xf_nstep': [], 'train_Yf_1step': [],
                                          'train_Yf_nstep': [], 'valid_Xf_1step': [], 'valid_Xf_nstep': [],
                                          'valid_Yf_1step': [], 'valid_Yf_nstep': [], 'test_Xf_1step': [],
                                          'test_Xf_nstep': [], 'test_Yf_1step': [], 'test_Yf_nstep': []}
    for COND,data_index in itertools.product(ALL_CONDITIONS, ls_data_indices):
        # Figure out if the index belongs to train, test or validation
        if data_index in ls_train_indices:
            key2_start = 'train_'
        elif data_index in ls_valid_indices:
            key2_start = 'valid_'
        else:
            key2_start = 'test_'

        # --- *** Generate prediction *** ---
        # Xf - 1 step
        psiXpT = dict_params['psixpT'].eval(feed_dict ={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})
        psiXfT_hat = np.matmul(psiXpT,dict_params['KxT_num'])
        XfT_hat = oc.inverse_transform_X(psiXfT_hat[:,0:n_genes],SYSTEM_NO)
        dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(
            r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfT_hat, multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(np.mean(np.square(XfT_hat)))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(
        #     r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfT_hat[:,0:n_genes], multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(
        #     r2_score(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), psiXfT_hat,
        #              multioutput='variance_weighted'))
        # Xf - n step
        psiXfTn_hat = psiXpT[0:1,:] # get the initial condition
        for i in range(len(dict_scaled_data[COND][data_index]['XfT'])): # predict n - steps
            psiXfTn_hat = np.concatenate([psiXfTn_hat, np.matmul(psiXfTn_hat[-1:],dict_params['KxT_num'])], axis = 0)
        psiXfTn_hat = psiXfTn_hat[1:,:]
        # Remove the initial condition and the lifted states; then unscale the variables
        XfTn_hat = oc.inverse_transform_X(psiXfTn_hat[:, 0:n_genes], SYSTEM_NO)
        dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(
            r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfTn_hat, multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(np.mean(np.square(XfTn_hat)))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(
        #     r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfTn_hat[:, 0:n_genes],multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(
        #     r2_score(dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}),
        #                                psiXfTn_hat, multioutput='variance_weighted'))
        # --- *** Compute the stats *** --- [for training, validation and test data sets separately]
        if with_output:
            # Yf - 1 step
            Yf_hat = oc.inverse_transform_Y(np.matmul(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), dict_params['WhT_num']),SYSTEM_NO)
            dict_instant_run_result[COND][key2_start + 'Yf_1step'].append(
                r2_score(dict_unscaled_data[COND][data_index]['YfT'].reshape(-1), XfT_hat.reshape(-1)))
            # Yf - n step
            Yfn_hat = oc.inverse_transform_Y(np.matmul(psiXfTn_hat,dict_params['WhT_num']), SYSTEM_NO)
            dict_instant_run_result[COND][key2_start + 'Yf_nstep'].append(
                r2_score(dict_unscaled_data[COND][data_index]['YfT'].reshape(-1), Yfn_hat.reshape(-1)))
        else:
            dict_instant_run_result[COND][key2_start + 'Yf_1step'].append(np.nan)
            dict_instant_run_result[COND][key2_start + 'Yf_nstep'].append(np.nan)
    # Save the stats to the dictionary - for MX,MN and NC, we save (train, test, valid) * (Xf1step, Xfnstep, Yf1step, Yfnstep)
    for COND in dict_instant_run_result.keys():
        for items in dict_instant_run_result[COND].keys():
            dict_instant_run_result[COND][items] =  np.mean(dict_instant_run_result[COND][items])
    dict_predict_STATS[run] = pd.DataFrame(dict_instant_run_result).T
    tf.reset_default_graph()
    sess.close()


for run in dict_predict_STATS.keys():
    print('=====================================================================')
    print('RUN: ', run)
    # print(dict_predict_STATS[run].loc[:,['train_Xf_1step', 'train_Xf_nstep', 'valid_Xf_1step', 'valid_Xf_nstep', 'test_Xf_1step', 'test_Xf_nstep',]])#, 'train_Yf_1step', 'train_Yf_nstep']])
    print(dict_predict_STATS[run].loc[:,
          ['train_Xf_1step', 'train_Xf_nstep', 'valid_Xf_1step', 'valid_Xf_nstep', 'test_Xf_1step',
           'test_Xf_nstep', 'train_Yf_1step', 'train_Yf_nstep']])
    print('=====================================================================')

# # TODO - Save the dictionary file
# with open(dict_predict_STATS_file, 'wb') as handle:
#      pickle.dump(dict_predict_STATS,handle)
for run in dict_predict_STATS.keys():
    print('RUN: ', run, ' r2_X_n-step_valid:', dict_predict_STATS[run].loc[:,'valid_Xf_nstep'].mean(),' r2_X_n-step_test:',dict_predict_STATS[run].loc[:,'test_Xf_nstep'].mean())
# TODO - predict the optimal run and save the best run
# dict_1step_results= {}
# dict_nstep_results= {}
# for run,COND in itertools.product(dict_predict_STATS.keys(),ALL_CONDITIONS):
# dict_1step_results[run] =


dict_resultable_1 = {}
for run in dict_predict_STATS.keys():
    with open(root_run_file + '/Sequential/Run_' + str(run) + '/dict_hyperparameters.pickle','rb') as handle:
        dict_hp = pickle.load(handle)
    dict_resultable_1[run] = {'run_no': run, 'lambda': dict_hp['regularization factor'], 'x_obs': dict_hp['x_obs'],
                              'n_l & n_n': [dict_hp['x_layers'], dict_hp['x_nodes']], ' r2_X_n_step_valid':
                             dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(), ' r2_X_n_step_test':
                             dict_predict_STATS[run].loc[:, 'test_Xf_nstep'].mean()}
df_resultable1 = pd.DataFrame(dict_resultable_1).T.sort_values(by ='x_obs')
print(df_resultable1)

plt.figure()
plt.plot(df_resultable1.loc[:,'x_obs'], df_resultable1.iloc[:,-2],'.')
plt.plot(df_resultable1.loc[:,'x_obs'], df_resultable1.iloc[:,-1],'.')
plt.xlabel('Number of observables')
plt.ylabel('$r^2$ for validation and test datasets')
plt.legend(['validation data','test data'])
plt.xlim([-0.5,2.5])
plt.ylim([0.98,1.001])
plt.show()

## OPTIMIZATION PROBLEM 1 - Save the best result 1
# SYSTEM_NO = 400
# RUN_NO = 3
# sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
# with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
#     d = pickle.load(handle)
# with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
#     d1 = pickle.load(handle)
# for items in d1.keys():
#     d[items] = d1[items]
# print(d.keys())
# with open('/Users/shara/Desktop/oc_deepDMD/System_'+str(SYSTEM_NO)+'_BestRun_1.pickle','wb') as handle:
#     pickle.dump(d,handle)

# TODO - Finish the run for the output fits today before sleep + start the runs for extended observables by tonight

## Do outputs lie in the span of the observables ?

run = 3 #18
print('RUN: ', run)
sess = tf.InteractiveSession()
run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
saver = tf.compat.v1.train.import_meta_graph(
    run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
dict_params = {}
dict_params['psixpT'] = tf.get_collection('psixpT')[0]
dict_params['psixfT'] = tf.get_collection('psixfT')[0]
dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])


# psiXpT_train = np.empty(shape=(0,len(psiXpT[0])))
Yps_train = np.empty(shape=(0,len(dict_scaled_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['YpT'][0])))
Yp_train = np.empty(shape=(0,len(dict_scaled_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['YpT'][0])))
for COND,data_index in itertools.product(ALL_CONDITIONS,ls_train_indices):
    try:
        psiXpT_train = np.concatenate([psiXpT_train,dict_params['psixpT'].eval(feed_dict ={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})],axis=0)
    except:
        psiXpT_train = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})
    Yps_train = np.concatenate([Yps_train, dict_scaled_data[COND][data_index]['YpT']], axis=0)
    Yp_train = np.concatenate([Yp_train, dict_unscaled_data[COND][data_index]['YpT']], axis=0)

lin_model_Y = LinearRegression(fit_intercept=False).fit(psiXpT_train,Yps_train)
print(r2_score(Yps_train.reshape(-1),lin_model_Y.predict(psiXpT_train).reshape(-1)))
print(r2_score(Yp_train.reshape(-1),oc.inverse_transform_Y(lin_model_Y.predict(psiXpT_train),SYSTEM_NO).reshape(-1)))

dict_indices = {'Train': ls_train_indices, 'Validation': ls_valid_indices, 'Test': ls_test_indices}
for items in dict_indices.keys():
    print('===============================================')
    print(items)
    print('===============================================')
    for COND,data_index in itertools.product(ALL_CONDITIONS,dict_indices[items]):
        print('COND: ', COND, 'Curve: ', data_index)
        psiXpT = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})
        Yp_hat = oc.inverse_transform_Y(lin_model_Y.predict(psiXpT),SYSTEM_NO)
        print('Fit Score r2 = ',r2_score(dict_unscaled_data[COND][data_index]['YpT'].reshape(-1), Yp_hat.reshape(-1)) )

tf.reset_default_graph()
sess.close()


## Checking Results of Run 2




