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


## OC deepDMD runs

# Preprocessing files
# SYSTEM_NO = 402
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(0,27)) # SYSTEM 402
# SYSTEM_NO = 404
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(0,27)) # SYSTEM 404
# SYSTEM_NO = 406
# ALL_CONDITIONS = ['MX']
# # ls_runs1 = list(range(0,72)) # SYSTEM 406 - Sequential
# ls_runs1 = list(range(0,72)) # SYSTEM 406 - Direct deepDMD
# SYSTEM_NO =408
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(0,32)) # SYSTEM 408
# SYSTEM_NO =409
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(40,50)) # SYSTEM 408
# SYSTEM_NO =410
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(0,50)) # SYSTEM 408
SYSTEM_NO = 416 #410,411,412
ALL_CONDITIONS = ['MX']
ls_runs1 = list(range(0,50)) # SYSTEM 408

# SYSTEM_NO = 500
# ALL_CONDITIONS = ['MX','MN']
# ls_runs1 = list(range(0,32)) # SYSTEM 406

# SYSTEM_NO = 501
# ALL_CONDITIONS = ['MX','MN']
# ls_runs1 = list(range(0,30)) # SYSTEM 406

# SYSTEM_NO = 601
# ALL_CONDITIONS = ['MX','MN']
# # ls_runs1 = list(range(0,72)) # SYSTEM 406 - Sequential
# ls_runs1 = list(range(0,72)) # SYSTEM 406 - Direct deepDMD


# METHOD = 'Sequential'
METHOD = 'deepDMD'

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

n_genes = len(dict_data_original[ALL_CONDITIONS[0]][ls_data_indices[0]]['df_X_TPM'])

dict_empty_all_conditions ={}
for COND in ALL_CONDITIONS:
    dict_empty_all_conditions[COND] = {}

dict_scaled_data = copy.deepcopy(dict_empty_all_conditions)
dict_unscaled_data = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    dict_intermediate = oc.scale_data_using_existing_scaler_folder(
        {'Xp': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
         'Xf': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T}, SYSTEM_NO)
    dict_scaled_data[COND][i] = {'XpT': dict_intermediate['Xp'], 'XfT': dict_intermediate['Xf']}
    dict_unscaled_data[COND][i] = {'XpT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                   'XfT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T}

# Generate predictions for each curve and write down the error statistics for each run
ls_all_run_indices = []
for folder in os.listdir(root_run_file + '/' + METHOD):
    if folder[0:4] == 'RUN_':  # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
ls_runs1 = set(ls_runs1).intersection(set(ls_all_run_indices))
# Open the predictions folder or create one if it doesn't exist
try:
    with open(dict_predict_STATS_file, 'rb') as handle:
        dict_predict_STATS = pickle.load(handle)
except:
    dict_predict_STATS = {}

dict_resultable1 ={}
# Generate the predictions for each run
for run in ls_runs1:
    dict_resultable1[run] = {}
    print('RUN: ', run)
    sess = tf.InteractiveSession()
    run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    dict_params['psixpT'] = tf.get_collection('psixpT')[0]
    dict_params['psixfT'] = tf.get_collection('psixfT')[0]
    dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
    dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
    dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
    dict_instant_run_result = copy.deepcopy(dict_empty_all_conditions)
    for items in dict_instant_run_result.keys():
        dict_instant_run_result[items] = {'train_Xf_1step': [], 'train_Xf_nstep': [], 'valid_Xf_1step': [], 'valid_Xf_nstep': [], 'test_Xf_1step': [], 'test_Xf_nstep': []}
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
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfT_hat))#, multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(np.mean(np.square(XfT_hat)))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfT_hat[:,0:n_genes], multioutput='variance_weighted'))
        dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), psiXfT_hat,multioutput='variance_weighted'))

        # Xf - n step
        psiXfTn_hat = psiXpT[0:1,:] # get the initial condition
        for i in range(len(dict_scaled_data[COND][data_index]['XfT'])): # predict n - steps
            psiXfTn_hat = np.concatenate([psiXfTn_hat, np.matmul(psiXfTn_hat[-1:],dict_params['KxT_num'])], axis = 0)
        psiXfTn_hat = psiXfTn_hat[1:,:]
        # Remove the initial condition and the lifted states; then unscale the variables
        XfTn_hat = oc.inverse_transform_X(psiXfTn_hat[:, 0:n_genes], SYSTEM_NO)
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfTn_hat))#, multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(np.mean(np.square(XfTn_hat)))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfTn_hat[:, 0:n_genes],multioutput='variance_weighted'))
        dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}),psiXfTn_hat, multioutput='variance_weighted'))

        # --- *** Compute the stats *** --- [for training, validation and test data sets separately]
    # Save the stats to the dictionary - for MX,MN and NC, we save (train, test, valid) * (Xf1step, Xfnstep, Yf1step, Yfnstep)
    for COND in dict_instant_run_result.keys():
        for items in dict_instant_run_result[COND].keys():
            dict_instant_run_result[COND][items] =  np.mean(dict_instant_run_result[COND][items])
    dict_predict_STATS[run] = pd.DataFrame(dict_instant_run_result).T
    dict_resultable1[run]['train_Xf_1step'] = dict_predict_STATS[run].loc[:,'train_Xf_1step'].mean()
    dict_resultable1[run]['valid_Xf_1step'] = dict_predict_STATS[run].loc[:,'valid_Xf_1step'].mean()
    dict_resultable1[run]['test_Xf_1step'] = dict_predict_STATS[run].loc[:,'test_Xf_1step'].mean()
    dict_resultable1[run]['train_Xf_nstep'] = dict_predict_STATS[run].loc[:,'train_Xf_nstep'].mean()
    dict_resultable1[run]['valid_Xf_nstep'] = dict_predict_STATS[run].loc[:,'valid_Xf_nstep'].mean()
    dict_resultable1[run]['test_Xf_nstep'] = dict_predict_STATS[run].loc[:,'test_Xf_nstep'].mean()
    tf.reset_default_graph()
    sess.close()

print('============================================================================')
print('RESULT TABLE 1')
df_resultable1 = pd.DataFrame(dict_resultable1).T
print(df_resultable1)
print('============================================================================')

# Need a plot of the results


dict_resultable_2 = {}
for run in dict_predict_STATS.keys():
    with open(root_run_file + '/' + METHOD+ '/Run_' + str(run) + '/dict_hyperparameters.pickle','rb') as handle:
        dict_hp = pickle.load(handle)
    dict_resultable_2[run] = { 'x_obs': dict_hp['x_obs'],
                              'n_l & n_n': [dict_hp['x_layers'], dict_hp['x_nodes']],  'r2_X_nstep_train':
                             dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(),'r2_X_nstep_valid':
                             dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(), 'r2_X_nstep_test':
                             dict_predict_STATS[run].loc[:, 'test_Xf_nstep'].mean(),'lambda': dict_hp['regularization factor']}
df_resultable2 = pd.DataFrame(dict_resultable_2).T.sort_values(by ='x_obs')
print('============================================================================')
print('RESULT TABLE 2')
print(df_resultable2)
print('============================================================================')

## OPTIMIZATION PROBLEM 1 - Save the best result 1
# # SYSTEM_NO = 401
# # RUN_NO = 4
# SYSTEM_NO = 402
# RUN_NO = 9 #25
# SYSTEM_NO = 406
# RUN_NO = 2
# sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
# run_folder_name = sys_folder_name + '/' + METHOD + '/RUN_' + str(RUN_NO)
# with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
#     d = pickle.load(handle)
# with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
#     d1 = pickle.load(handle)
# for items in d1.keys():
#     d[items] = d1[items]
# # print(d.keys())
# with open('/Users/shara/Desktop/oc_deepDMD/System_'+str(SYSTEM_NO)+'_BestRun_1.pickle','wb') as handle:
#     pickle.dump(d,handle)

# ## TESTING OUT OUTPUT
# run = 2
# sess = tf.InteractiveSession()
# run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
# saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
# saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
# dict_params = {}
# dict_params['psixpT'] = tf.get_collection('psixpT')[0]
# dict_params['psixfT'] = tf.get_collection('psixfT')[0]
# dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
# dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
# dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
#
dict_ALLDATA = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ALL_CONDITIONS)
#
#
# for phase in ['train','valid','test']:
#     psiXfTs = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_ALLDATA[phase]['XfTs']})
#     YfTs = dict_ALLDATA[phase]['YfTs']
#     if phase == 'train':
#         model_Y = LinearRegression(fit_intercept=True)
#         model_Y.fit(psiXfTs, YfTs)
#     print(phase, 'score [scaled]: ', r2_score(YfTs.reshape(-1),model_Y.predict(psiXfTs).reshape(-1)))
#     print(phase, 'score [unscaled]: ', r2_score(dict_ALLDATA['Y_scaler'].inverse_transform(dict_ALLDATA[phase]['YfTs']).reshape(-1),dict_ALLDATA['Y_scaler'].inverse_transform(model_Y.predict(psiXfTs)).reshape(-1)))
# tf.reset_default_graph()
# sess.close()

## Output predictions
#
# # SYSTEM_NO = 402
# # ALL_CONDITIONS = ['MX']
# # ls_runs1 = list(range(28,36)) # SYSTEM 402
# SYSTEM_NO = 406
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(8,16)) # SYSTEM 402
#
# Generate predictions for each curve and write down the error statistics for each run
ls_all_run_indices = []
for folder in os.listdir(root_run_file + '/' + METHOD):
    if folder[0:4] == 'RUN_':  # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
ls_runs1 = set(ls_runs1).intersection(set(ls_all_run_indices))
dict_predict_STATS_Y = {}

dict_resultable1_Y ={}
# Generate the predictions for each run
for run in ls_runs1:
    dict_resultable1_Y[run] = {}
    print('RUN: ', run)
    sess = tf.InteractiveSession()
    run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
    saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    dict_params['psixfT'] = tf.get_collection('psixfT')[0]
    dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
    dict_params['WhT_num'] = sess.run(tf.get_collection('WhT')[0])
    dict_instant_run_result = copy.deepcopy(dict_empty_all_conditions)
    for items in dict_instant_run_result.keys():
        dict_instant_run_result[items] = {'train_Yf': [], 'valid_Yf': [], 'test_Yf': []}
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
        psiXpT = dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: dict_ALLDATA['scaled'][COND][data_index]['XfT']})
        YfTs_hat = np.matmul(psiXpT,dict_params['WhT_num'])
        YfT_hat = oc.inverse_transform_Y(YfTs_hat,SYSTEM_NO)
        dict_instant_run_result[COND][key2_start + 'Yf'].append(r2_score(dict_ALLDATA['unscaled'][COND][data_index]['YfT'].reshape(-1), YfT_hat.reshape(-1)))
        # --- *** Compute the stats *** --- [for training, validation and test data sets separately]
    # Save the stats to the dictionary - for MX,MN and NC, we save (train, test, valid) * (Xf1step, Xfnstep, Yf1step, Yfnstep)
    for COND in dict_instant_run_result.keys():
        for items in dict_instant_run_result[COND].keys():
            dict_instant_run_result[COND][items] =  np.mean(dict_instant_run_result[COND][items])
    dict_predict_STATS_Y[run] = pd.DataFrame(dict_instant_run_result).T
    dict_resultable1_Y[run]['train_Yf'] = dict_predict_STATS_Y[run].loc[:,'train_Yf'].mean()
    dict_resultable1_Y[run]['valid_Yf'] = dict_predict_STATS_Y[run].loc[:,'valid_Yf'].mean()
    dict_resultable1_Y[run]['test_Yf'] = dict_predict_STATS_Y[run].loc[:,'test_Yf'].mean()
    tf.reset_default_graph()
    sess.close()

print('============================================================================')
print('RESULT TABLE 1')
df_resultable1_Y = pd.DataFrame(dict_resultable1_Y).T
print(df_resultable1_Y)
print('============================================================================')
dict_resultable_2_Y = {}
for run in dict_predict_STATS_Y.keys():
    with open(root_run_file + '/' + METHOD + '/Run_' + str(run) + '/dict_hyperparameters.pickle','rb') as handle:
        dict_hp = pickle.load(handle)
    if METHOD == 'Sequential':
        dict_resultable_2_Y[run] = {'y_obs': dict_hp['y_obs'],
                                  'n_l & n_n': [dict_hp['y_layers'], dict_hp['y_nodes']],  ' r2_Yf_train':
                                 dict_predict_STATS_Y[run].loc[:, 'train_Yf'].mean(),' r2_Yf_valid':
                                 dict_predict_STATS_Y[run].loc[:, 'valid_Yf'].mean(), ' r2_Yf_test':
                                 dict_predict_STATS_Y[run].loc[:, 'test_Yf'].mean(),'lambda': dict_hp['regularization factor']}
    elif METHOD == 'deepDMD':
        dict_resultable_2_Y[run] = {'x_obs': dict_hp['x_obs'],
                                    'n_l & n_n': [dict_hp['x_layers'], dict_hp['x_nodes']], 'r2_Yf_train':
                                        dict_predict_STATS_Y[run].loc[:, 'train_Yf'].mean(), 'r2_Yf_valid':
                                        dict_predict_STATS_Y[run].loc[:, 'valid_Yf'].mean(), 'r2_Yf_test':
                                        dict_predict_STATS_Y[run].loc[:, 'test_Yf'].mean(),
                                    'lambda': dict_hp['regularization factor']}
if METHOD == 'Sequential':
    df_resultable2_Y = pd.DataFrame(dict_resultable_2_Y).T.sort_values(by='y_obs')
elif METHOD == 'deepDMD':
    df_resultable2_Y = pd.DataFrame(dict_resultable_2_Y).T.sort_values(by='x_obs')
print('============================================================================')
print('RESULT TABLE 2')
print(df_resultable2_Y)
print('============================================================================')




# ## RESULTS PLOT =========================================================================================================
#
# # Plotting the eigenfunctions
#
# run = 29
# sess = tf.InteractiveSession()
# run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
# saver = tf.compat.v1.train.import_meta_graph(run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
# saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
# dict_params = {}
# dict_params['psixpT'] = tf.get_collection('psixpT')[0]
# dict_params['psixfT'] = tf.get_collection('psixfT')[0]
# dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
# dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
# dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
# Kx = dict_params['KxT_num'].T
# e_in,W_in = np.linalg.eig(Kx)
# E_in = np.diag(e_in)
# E, W, comp_modes, comp_modes_conj = rnaf.resolve_complex_right_eigenvalues(copy.deepcopy(E_in), copy.deepcopy(W_in))
# Winv = np.linalg.inv(W)
#
# # Plot for the K
# Kx = dict_params['KxT_num'].T
# E_complex = np.linalg.eigvals(Kx)
# # K matrix heatmap
# plt.figure(figsize=(12,10))
# a = sb.heatmap(Kx, cmap="RdYlGn",center=0,vmax=np.abs(Kx).max(),vmin=-np.abs(Kx).max())
# b, t = a.axes.get_ylim()  # discover the values for bottom and top
# b += 0.5  # Add 0.5 to the bottom
# t -= 0.5  # Subtract 0.5 from the top
# cbar = a.collections[0].colorbar
# ls_gene_tags = list(dict_data_original['MX'][0]['df_X_TPM'].index)
# p = rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags = ls_gene_tags,search_columns='genes(OLN),genes(PREFERRED)')
# ls_genes = []
# for i in range(len(ls_gene_tags)):
#     if p[p['Gene names  (ordered locus )']==ls_gene_tags[i]].iloc[0,1] == '':
#         ls_genes.append(p[p['Gene names  (ordered locus )']==ls_gene_tags[i]].iloc[0, 0])
#     else:
#         ls_genes.append(p[p['Gene names  (ordered locus )'] == ls_gene_tags[i]].iloc[0, 1])
# for i in range(len(Kx) - len(ls_genes)-1):
#     ls_genes.append('$\\varphi_{{{}}}(x)$'.format(i+1))
# ls_genes.append('$\\varphi_{0}(x)$')
# a.set_xticks(np.arange(0.5,len(ls_genes),1))
# a.set_yticks(np.arange(0.5,len(ls_genes),1))
# a.set_xticklabels(ls_genes,rotation =90,fontsize =19)
# a.set_yticklabels(ls_genes,rotation =0,fontsize =19)
# a.axes.set_ylim(b, t)
# # here set the labelsize by 20
# # cbar.ax.tick_params(labelsize=FONTSIZE)
# # a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
# # a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
# plt.show()
#
#
# dict_growth_genes_biocyc = rnaf.get_PputidaKT2440_growth_genes()
# for i in ls_gene_tags:
#     if i in dict_growth_genes_biocyc['cell_cycle']:
#         print('Gene ',i, ' is in biocyc cell cycle list')
#     elif i in dict_growth_genes_biocyc['cell_division']:
#         print('Gene ',i, ' is in biocyc cell division list')
#     else:
#         print('Gene ', i, ' is not in biocyc list')
#
#
# # Eigenvalue plot
# fig = plt.figure(figsize=(7,7))
# ax = fig.add_subplot(1, 1, 1)
# circ = plt.Circle((0, 0), radius=1, edgecolor='None', facecolor='cyan')
# ax.add_patch(circ)
# ax.plot(np.real(E_complex),np.imag(E_complex),'x',linewidth=5,color='g',markersize =12)
# ax.set_xlabel('$Re(\lambda)$')
# ax.set_ylabel('$Im(\lambda)$')
# ax.set_xticks([-1.0,-0.5,0,0.5,1.0])
# ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
# ax.set_xlim([-1.1,1.1])
# ax.set_ylim([-1.1,1.1])
# plt.show()
#
# #
# index = 0
# n_funcs = 8
# XT = np.concatenate([dict_ALLDATA['unscaled']['MX'][index]['XpT'][0:1,:],dict_ALLDATA['unscaled']['MX'][index]['XfT']],axis=0)
# XTs = dict_ALLDATA['X_scaler'].transform(XT)
# psiXTs_true = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: XTs})
# psiXTs = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_ALLDATA['scaled']['MX'][index]['XpT'][0:1,:]})
# for i in range(len(dict_ALLDATA['unscaled']['MX'][index]['XfT'])):
#     psiXTs = np.concatenate([psiXTs,np.matmul(psiXTs[-1:],dict_params['KxT_num'])])
# Phis = np.matmul(Winv, psiXTs.T)
# # Phis = np.matmul(Winv, psiXTs_true.T)
# YT = np.concatenate([dict_ALLDATA['unscaled']['MX'][index]['YpT'][0:1,:],dict_ALLDATA['unscaled']['MX'][index]['YfT']],axis=0)
# YTs = dict_ALLDATA['Y_scaler'].transform(YT)
#
# x_ticks = np.array([1,2,3,4,5,6,7])
# f,ax_o = plt.subplots(np.int(np.ceil(n_funcs/2)),2,sharex=True,figsize=(10,n_funcs*1.5))
# # f,ax = plt.subplots(n_funcs,1,sharex=True,figsize=(5,n_funcs*1.5))
# ax = ax_o.reshape(-1)
# eig_func_index = 0
# for i in range(n_funcs):
#     try:
#         if eig_func_index in comp_modes:
#             ax[i].plot(x_ticks, Phis[eig_func_index, :],label = 'Real')
#             ax[i].plot(x_ticks, Phis[eig_func_index+1, :],label = 'Imaginary')
#             # ax[i].legend()
#             real_part = round(np.abs(E[eig_func_index,eig_func_index]),3)
#             imag_part = round(np.abs(E[eig_func_index,eig_func_index+1]),3)
#             ax[i].set_title('$\phi_{{{},{}}}(x)$'.format(eig_func_index+1,eig_func_index+2) + ', $\lambda =$' + str(real_part) +'$\pm$j' +  str(imag_part))
#             eig_func_index = eig_func_index + 2
#         else:
#             ax[i].plot(x_ticks, Phis[eig_func_index, :])
#             real_part = round(np.abs(E[eig_func_index, eig_func_index]), 3)
#             # print(real_part)
#             # ax[i].set_title('$\phi_{{{}}}(x), \lambda = {{{}}}$'.format(eig_func_index+1,real_part))
#             ax[i].set_title('$\phi_{{{}}}(x)$'.format(eig_func_index + 1) + ', $\lambda = $'+ str(real_part))
#             eig_func_index = eig_func_index + 1
#     except:
#         break
# ax_o[-1,0].set_xlabel('time (hrs)')
# ax_o[-1,1].set_xlabel('time (hrs)')
# ax_o[-1,0].set_xticks([0,3,6])
# ax_o[-1,1].set_xticks([0,3,6])
# f.show()
#
# ls_gene_max_var_index = sorted(range(len(XT.var(axis=0))), key=lambda i: XT.var(axis=0)[i])
# ls_gene_max_var_index.reverse()
# plt.figure(figsize=(10,6))
# for i in range(n_funcs):
#     try:
#         plt.plot(x_ticks,XTs[:,ls_gene_max_var_index[i]],label = 'gene_' + str(i))
#     except:
#         break
# plt.legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=4)
# plt.show()
#
# plt.figure(figsize=(10,6))
# for i in range(len(psiXTs[0])-len(XTs[0])):
#     if i == len(psiXTs[0]):
#         plt.plot(x_ticks, psiXTs[:,len(XTs[0])+i ], label='$\phi_{0}(x)$')
#     else:
#         plt.plot(x_ticks,psiXTs[:,len(XTs[0])+i],label = '$\phi_{{{}}}(x)$'.format(i + 1) )
# plt.legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=4)
# plt.show()
#
#
# print(r2_score(psiXTs_true,psiXTs,multioutput='raw_values'))
#
# # dict_x_index ={'MX': np.array([2,3,4,5,6,7]),'MN': np.array([4,5,6,7]),'NC': np.array([4,5,6,7])}
# # for COND_NO in range(len(ALL_CONDITIONS)):
# #     COND = ALL_CONDITIONS[COND_NO]
# #     Xs = dict_ALLDATA['X_scaler'].transform(X)
# #     Phis = np.matmul(Winv, Xps)
# #     Phis = Phis[0:-1,:]
# #     Phi = oc.inverse_transform_X(Phis.T,SYSTEM_NO).T
# #     for i in range(n_funcs):
# #         if i==0:
# #             ax[i].plot(dict_x_index[COND], Phi[i, :],label = COND)
# #         else:
# #             ax[i].plot(dict_x_index[COND], Phi[i,:])
# #         ax[i].set_title('$\lambda = $'+ str(E[i][i]))
# #
# # ax[0].legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
# # ax[-1].set_xlabel('Time [hrs]')
# # f.show()
#
#
# tf.reset_default_graph()
# sess.close()
#
# ##
# # run = 13
#
# sess = tf.InteractiveSession()
# run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
# saver = tf.compat.v1.train.import_meta_graph(
#     run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
# saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
# dict_params = {}
# dict_params['psixfT'] = tf.get_collection('psixfT')[0]
# dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
# dict_params['WhT_num'] = sess.run(tf.get_collection('WhT')[0])
#
#
# # Wh matrix heatmap
# Wh = dict_params['WhT_num'].T
# plt.figure(figsize=(20,20))
# a = sb.heatmap(Wh, cmap="RdYlGn",center=0,vmax=np.abs(Wh).max(),vmin=-np.abs(Wh).max())
# b, t = a.axes.get_ylim()  # discover the values for bottom and top
# b += 0.5  # Add 0.5 to the bottom
# t -= 0.5  # Subtract 0.5 from the top
# a.axes.set_ylim(b, t)
# cbar = a.collections[0].colorbar
# # here set the labelsize by 20
# # cbar.ax.tick_params(labelsize=FONTSIZE)
# # a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
# # a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
# plt.show()
#
#
# B = np.matmul(Wh,W)
# plt.figure(figsize=(20,20))
# a = sb.heatmap(B, cmap="RdYlGn",center=0,vmax=np.abs(B).max(),vmin=-np.abs(B).max())
# b, t = a.axes.get_ylim()  # discover the values for bottom and top
# b += 0.5  # Add 0.5 to the bottom
# t -= 0.5  # Subtract 0.5 from the top
# a.axes.set_ylim(b, t)
# cbar = a.collections[0].colorbar
# # here set the labelsize by 20
# # cbar.ax.tick_params(labelsize=FONTSIZE)
# # a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
# # a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
# plt.show()
#
# Bmean = B.mean(axis=0)
# ls_modes_ordered_by_output = sorted(range(len(np.abs(Bmean))), key=lambda i: np.abs(Bmean)[i])
# ls_modes_ordered_by_output.reverse()
#
# x_ticks = np.array([1,2,3,4,5,6,7])
# f,ax = plt.subplots(5,2,sharex=True,figsize=(10,n_funcs*1.5))
# ax = ax.reshape(-1)
# eig_func_index = 0
# for i in range(n_funcs):
#     try:
#         # if eig_func_index in comp_modes:
#         #     ax[i].plot(x_ticks, Phis[i, :],label = 'Real')
#         #     ax[i].plot(x_ticks, Phis[i+1, :],label = 'Imaginary')
#         #     # ax[i].legend()
#         #     real_part = round(np.abs(E[eig_func_index,eig_func_index]),3)
#         #     imag_part = round(np.abs(E[eig_func_index,eig_func_index+1]),3)
#         #     ax[i].set_title('$\phi_{{{},{}}}(x)$'.format(eig_func_index+1,eig_func_index+2) + ', $\lambda =$' + str(real_part) +'$\pm$j' +  str(imag_part))
#         #     eig_func_index = eig_func_index + 2
#         # else:
#         ax[i].plot(x_ticks, Phis[ls_modes_ordered_by_output[i], :])
#         real_part = round(np.abs(E[ls_modes_ordered_by_output[i], ls_modes_ordered_by_output[i]]), 3)
#         # print(real_part)
#         # ax[i].set_title('$\phi_{{{}}}(x), \lambda = {{{}}}$'.format(eig_func_index+1,real_part))
#         ax[i].set_title('$\phi_{{{}}}(x)$'.format(eig_func_index + 1) + ', $\lambda = $'+ str(real_part))
#         eig_func_index = eig_func_index + 1
#     except:
#         break
# f.show()
#
# YT = np.concatenate([dict_ALLDATA['unscaled']['MX'][index]['YpT'][0:1,:],dict_ALLDATA['unscaled']['MX'][index]['YfT']],axis=0)
# YTs = dict_ALLDATA['Y_scaler'].transform(YT)
# plt.figure()
# plt.plot(x_ticks,YTs)
# plt.show()
#
#
#
# tf.reset_default_graph()
# sess.close()


# Training result plot

ls_obs = list(df_resultable2.loc[:,'x_obs'].unique())
# ls_obs=[0,1,2,3,4]
dict_result3 = {}
for i in ls_obs:
    df_temp = copy.deepcopy(df_resultable2[df_resultable2['x_obs'] ==i])
    dict_result3[i] = {}
    dict_result3[i]['r2_X_nstep_train_mean'] = np.maximum(0,df_temp.loc[:,'r2_X_nstep_train']).mean()
    dict_result3[i]['r2_X_nstep_train_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_train']).std()
    dict_result3[i]['r2_X_nstep_valid_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).mean()
    dict_result3[i]['r2_X_nstep_valid_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).std()
    dict_result3[i]['r2_X_nstep_test_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).mean()
    dict_result3[i]['r2_X_nstep_test_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).std()
    df_temp = copy.deepcopy(df_resultable2_Y[df_resultable2_Y['x_obs'] == i])
    dict_result3[i]['r2_Y_train_mean'] = np.maximum(0,df_temp.loc[:, 'r2_Yf_train']).mean()
    dict_result3[i]['r2_Y_train_std'] = np.maximum(0,df_temp.loc[:, 'r2_Yf_train']).std()
    dict_result3[i]['r2_Y_valid_mean'] = np.maximum(0,df_temp.loc[:, 'r2_Yf_valid']).mean()
    dict_result3[i]['r2_Y_valid_std'] = np.maximum(0,df_temp.loc[:, 'r2_Yf_valid']).std()
    dict_result3[i]['r2_Y_test_mean'] = np.maximum(0,df_temp.loc[:, 'r2_Yf_test']).mean()
    dict_result3[i]['r2_Y_test_std'] = np.maximum(0,df_temp.loc[:, 'r2_Yf_test']).std()

# df_results3_deepDMD = copy.deepcopy(pd.DataFrame(dict_result3).T)
df_results3_OCdeepDMD = copy.deepcopy(pd.DataFrame(dict_result3).T)

##
ls_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
plt.figure()
ls_obs_select =[0,1,2,3,4,5,6,7]
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_train_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_train_std'],label = 'OCdeepDMD xf train', capsize=9,color=ls_colors[0])
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_valid_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_valid_std'],label = 'OCdeepDMD xf valid', capsize=5,color=ls_colors[1])
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_test_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_test_std'],label = 'OCdeepDMD xf test', capsize=2,color=ls_colors[2])
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_Y_train_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_Y_train_std'],label = 'yf train', capsize=9,color=ls_colors[3])
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_Y_valid_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_Y_valid_std'],label = 'yf valid', capsize=5,color=ls_colors[4])
plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_Y_test_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_Y_test_std'],label = 'yf test', capsize=1,color=ls_colors[5])
plt.legend(ncol =2, fontsize=10)
plt.xlabel('$n_{observables}$')
plt.ylabel('$r^2$')
# plt.xlim([0,8])
plt.show()

##
# ls_colors =['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
# plt.figure()
# ls_obs_select =[0,1,2,3,6]
# plt.errorbar(ls_obs_select,df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_train_mean'],yerr=df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_train_std'],label = 'deepDMD xf train', capsize=9,linestyle='--',color=ls_colors[0])
# plt.errorbar(ls_obs_select,df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_valid_mean'],yerr=df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_valid_std'],label = 'deepDMD xf valid', capsize=5,linestyle='--',color=ls_colors[1])
# plt.errorbar(ls_obs_select,df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_test_mean'],yerr=df_results3_deepDMD.loc[ls_obs_select,'r2_X_nstep_test_std'],label = 'deepDMD xf test', capsize=2,linestyle='--',color=ls_colors[2])
# plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_train_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_train_std'],label = 'OCdeepDMD xf train', capsize=9,color=ls_colors[0])
# plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_valid_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_valid_std'],label = 'OCdeepDMD xf valid', capsize=5,color=ls_colors[1])
# plt.errorbar(ls_obs_select,df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_test_mean'],yerr=df_results3_OCdeepDMD.loc[ls_obs_select,'r2_X_nstep_test_std'],label = 'OCdeepDMD xf test', capsize=2,color=ls_colors[2])
# # plt.errorbar(ls_obs,df_results3.loc[:,'r2_Y_train_mean'],yerr=df_results3.loc[:,'r2_Y_train_std'],label = 'yf train', capsize=9)
# # plt.errorbar(ls_obs,df_results3.loc[:,'r2_Y_valid_mean'],yerr=df_results3.loc[:,'r2_Y_valid_std'],label = 'yf valid', capsize=5)
# # plt.errorbar(ls_obs,df_results3.loc[:,'r2_Y_test_mean'],yerr=df_results3.loc[:,'r2_Y_test_std'],label = 'yf test', capsize=1)
# plt.legend(ncol =2, fontsize=10)
# plt.xlabel('$n_{observables}$')
# plt.ylabel('$r^2$')
# # plt.xlim([0,8])
# plt.show()
