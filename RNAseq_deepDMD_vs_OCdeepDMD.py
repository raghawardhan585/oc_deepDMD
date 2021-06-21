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

SYSTEM_NO = 704 #410,411,412
ALL_CONDITIONS = ['MX']
ls_runs1 = list(range(0,60)) # SYSTEM 408

# df_results2_deepDMD = rnaf.generate_n_step_prediction_table(SYSTEM_NO,ALL_CONDITIONS=['MX'],ls_runs1=list(range(0,100)),METHOD = 'Sequential')
# ls_obs_deepDMD = list(df_results2_deepDMD.loc[:,'x_obs'].unique())
# # ls_obs=[0,1,2,3,4]
# dict_result3 = {}
# for i in ls_obs_deepDMD:
#     df_temp = copy.deepcopy(df_results2_deepDMD[df_results2_deepDMD['x_obs'] ==i])
#     dict_result3[i] = {}
#     dict_result3[i]['r2_X_nstep_train_mean'] = np.maximum(0,df_temp.loc[:,'r2_X_nstep_train']).mean()
#     dict_result3[i]['r2_X_nstep_train_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_train']).std()
#     dict_result3[i]['r2_X_nstep_valid_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).mean()
#     dict_result3[i]['r2_X_nstep_valid_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).std()
#     dict_result3[i]['r2_X_nstep_test_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).mean()
#     dict_result3[i]['r2_X_nstep_test_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).std()
# df_results3_deepDMD = copy.deepcopy(pd.DataFrame(dict_result3).T)


df_results2_OCdeepDMD = rnaf.generate_n_step_prediction_table(SYSTEM_NO,ALL_CONDITIONS=['MX'],ls_runs1=list(range(0,100)),METHOD = 'deepDMD')
ls_obs_OCdeepDMD = list(df_results2_OCdeepDMD.loc[:,'x_obs'].unique())
dict_result3 = {}
for i in ls_obs_OCdeepDMD:
    df_temp = copy.deepcopy(df_results2_OCdeepDMD[df_results2_OCdeepDMD['x_obs'] ==i])
    dict_result3[i] = {}
    dict_result3[i]['r2_X_nstep_train_mean'] = np.maximum(0,df_temp.loc[:,'r2_X_nstep_train']).mean()
    dict_result3[i]['r2_X_nstep_train_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_train']).std()
    dict_result3[i]['r2_X_nstep_valid_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).mean()
    dict_result3[i]['r2_X_nstep_valid_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).std()
    dict_result3[i]['r2_X_nstep_test_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).mean()
    dict_result3[i]['r2_X_nstep_test_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).std()
df_results3_OCdeepDMD = copy.deepcopy(pd.DataFrame(dict_result3).T)


##
ls_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure()
# ls_obs_select =[0,1,2,3,4,5,6,7]
# ls_obs_select = list(set(ls_obs_deepDMD).intersection(set(ls_obs_OCdeepDMD)))
ls_obs_select = ls_obs_OCdeepDMD
# plt.errorbar(ls_obs_select,df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_train_mean'],yerr=df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_train_std'],label = 'deepDMD xf train', capsize=9,linestyle='--',color=ls_colors[0],linewidth=1)
# plt.errorbar(ls_obs_select,df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_valid_mean'],yerr=df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_valid_std'],label = 'deepDMD xf valid', capsize=5,linestyle='--',color=ls_colors[1])
# plt.errorbar(ls_obs_select,df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_test_mean'],yerr=df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_test_std'],label = 'deepDMD xf test', capsize=2,linestyle='--',color=ls_colors[2])
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_train_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_train_std'],label = 'OCdeepDMD xf train', capsize=9,color=ls_colors[0],linewidth=2)
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_valid_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_valid_std'],label = 'OCdeepDMD xf valid', capsize=5,color=ls_colors[1],linewidth=2)
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_test_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_test_std'],label = 'OCdeepDMD xf test', capsize=2,color=ls_colors[2],linewidth=2)
plt.legend(ncol =2, fontsize=10)
plt.xlabel('$n_{observables}$')
plt.ylabel('$r^2$')
# plt.xlim([0,8])
plt.show()

##

# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 702,run =4,METHOD = 'deepDMD',ALL_CONDITIONS=['MX'])
# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 701,run =45, METHOD = 'deepDMD',ALL_CONDITIONS=['MX'])
# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 701,run =21, METHOD = 'Sequential',ALL_CONDITIONS=['MX'])

# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 702, run = 17, METHOD = 'deepDMD',ALL_CONDITIONS=['MX'])
# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 702,run = 0, METHOD = 'Sequential',ALL_CONDITIONS=['MX'])

# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 703, run = 26, METHOD = 'deepDMD',ALL_CONDITIONS=['MX'])
# rnaf.plot_dynamics_related_graphs(SYSTEM_NO = 703,run =3, METHOD = 'Sequential',ALL_CONDITIONS=['MX'])

# run =21
# root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
#         SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
# # Indices [train, validation and test]
# with open(indices_path, 'rb') as handle:
#     ls_data_indices = pickle.load(handle)
# ls_train_indices = ls_data_indices[0:12]
# ls_valid_indices = ls_data_indices[12:14]
# ls_test_indices = ls_data_indices[14:16]
# dict_temp = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS=['MX'])
# dict_scaled_data = dict_temp['scaled']
# dict_unscaled_data = dict_temp['unscaled']
#
# print('RUN: ', run)
# sess = tf.InteractiveSession()
# run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
# saver = tf.compat.v1.train.import_meta_graph(
#     run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
# saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
# dict_params = {}
# dict_params['psixpT'] = tf.get_collection('psixpT')[0]
# dict_params['psixfT'] = tf.get_collection('psixfT')[0]
# dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
# dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
# dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
#
#
# # psiXpT_train = np.empty(shape=(0,len(psiXpT[0])))
# Yps_train = np.empty(shape=(0,len(dict_scaled_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['YpT'][0])))
# Yp_train = np.empty(shape=(0,len(dict_scaled_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['YpT'][0])))
# for COND,data_index in itertools.product(ALL_CONDITIONS,ls_train_indices):
#     try:
#         psiXpT_train = np.concatenate([psiXpT_train,dict_params['psixpT'].eval(feed_dict ={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})],axis=0)
#     except:
#         psiXpT_train = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})
#     Yps_train = np.concatenate([Yps_train, dict_scaled_data[COND][data_index]['YpT']], axis=0)
#     Yp_train = np.concatenate([Yp_train, dict_unscaled_data[COND][data_index]['YpT']], axis=0)
#
# lin_model_Y = LinearRegression(fit_intercept=False).fit(psiXpT_train,Yps_train)
# print(r2_score(Yps_train.reshape(-1),lin_model_Y.predict(psiXpT_train).reshape(-1)))
# print(r2_score(Yp_train.reshape(-1),oc.inverse_transform_Y(lin_model_Y.predict(psiXpT_train),SYSTEM_NO).reshape(-1)))
#
# dict_indices = {'Train': ls_train_indices, 'Validation': ls_valid_indices, 'Test': ls_test_indices}
# for items in dict_indices.keys():
#     print('===============================================')
#     print(items)
#     print('===============================================')
#     for COND,data_index in itertools.product(ALL_CONDITIONS,dict_indices[items]):
#         print('COND: ', COND, 'Curve: ', data_index)
#         psiXpT = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})
#         Yp_hat = oc.inverse_transform_Y(lin_model_Y.predict(psiXpT),SYSTEM_NO)
#         print('Fit Score r2 = ',r2_score(dict_unscaled_data[COND][data_index]['YpT'].reshape(-1), Yp_hat.reshape(-1)) )
# tf.reset_default_graph()
# sess.close()


