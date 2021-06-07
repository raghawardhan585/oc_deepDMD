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
SYSTEM_NO = 402
ALL_CONDITIONS = ['MX']
# ALL_CONDITIONS = ['MX','MN']#list(dict_data_original.keys())
# ls_runs1 = list(range(0,60)) # SYSTEM 401
ls_runs1 = list(range(0,27)) # SYSTEM 402
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
for folder in os.listdir(root_run_file + '/Sequential'):
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
    run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
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
        dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfT_hat, multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(np.mean(np.square(XfT_hat)))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfT_hat[:,0:n_genes], multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), psiXfT_hat,multioutput='variance_weighted'))

        # Xf - n step
        psiXfTn_hat = psiXpT[0:1,:] # get the initial condition
        for i in range(len(dict_scaled_data[COND][data_index]['XfT'])): # predict n - steps
            psiXfTn_hat = np.concatenate([psiXfTn_hat, np.matmul(psiXfTn_hat[-1:],dict_params['KxT_num'])], axis = 0)
        psiXfTn_hat = psiXfTn_hat[1:,:]
        # Remove the initial condition and the lifted states; then unscale the variables
        XfTn_hat = oc.inverse_transform_X(psiXfTn_hat[:, 0:n_genes], SYSTEM_NO)
        dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfTn_hat, multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(np.mean(np.square(XfTn_hat)))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfTn_hat[:, 0:n_genes],multioutput='variance_weighted'))
        # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}),psiXfTn_hat, multioutput='variance_weighted'))

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
dict_resultable_2 = {}
for run in dict_predict_STATS.keys():
    with open(root_run_file + '/Sequential/Run_' + str(run) + '/dict_hyperparameters.pickle','rb') as handle:
        dict_hp = pickle.load(handle)
    dict_resultable_2[run] = { 'x_obs': dict_hp['x_obs'],
                              'n_l & n_n': [dict_hp['x_layers'], dict_hp['x_nodes']],  ' r2_X_n_train_valid':
                             dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(),' r2_X_n_step_valid':
                             dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(), ' r2_X_n_step_test':
                             dict_predict_STATS[run].loc[:, 'test_Xf_nstep'].mean(),'lambda': dict_hp['regularization factor']}
df_resultable2 = pd.DataFrame(dict_resultable_2).T.sort_values(by ='x_obs')
print('============================================================================')
print('RESULT TABLE 2')
print(df_resultable2)
print('============================================================================')

## OPTIMIZATION PROBLEM 1 - Save the best result 1
# SYSTEM_NO = 401
# RUN_NO = 4
SYSTEM_NO = 402
RUN_NO = 9 #25
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
run_folder_name = sys_folder_name + '/Sequential/RUN_' + str(RUN_NO)
with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
    d = pickle.load(handle)
with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
    d1 = pickle.load(handle)
for items in d1.keys():
    d[items] = d1[items]
# print(d.keys())
with open('/Users/shara/Desktop/oc_deepDMD/System_'+str(SYSTEM_NO)+'_BestRun_1.pickle','wb') as handle:
    pickle.dump(d,handle)

## TESTING OUT OUTPUT
run = 9
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

dict_ALLDATA = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ['MX'])


for phase in ['train','valid','test']:
    psiXfTs = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_ALLDATA[phase]['XfTs']})
    YfTs = dict_ALLDATA[phase]['YfTs']
    if phase == 'train':
        model_Y = LinearRegression(fit_intercept=True)
        model_Y.fit(psiXfTs, YfTs)
    print(phase, 'score [scaled]: ', r2_score(YfTs.reshape(-1),model_Y.predict(psiXfTs).reshape(-1)))
    print(phase, 'score [unscaled]: ', r2_score(dict_ALLDATA['Y_scaler'].inverse_transform(dict_ALLDATA[phase]['YfTs']).reshape(-1),dict_ALLDATA['Y_scaler'].inverse_transform(model_Y.predict(psiXfTs)).reshape(-1)))
tf.reset_default_graph()
sess.close()

## Output predictions

SYSTEM_NO = 402
ALL_CONDITIONS = ['MX']
ls_runs1 = list(range(28,36)) # SYSTEM 402

# Generate predictions for each curve and write down the error statistics for each run
ls_all_run_indices = []
for folder in os.listdir(root_run_file + '/Sequential'):
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
    run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
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
    with open(root_run_file + '/Sequential/Run_' + str(run) + '/dict_hyperparameters.pickle','rb') as handle:
        dict_hp = pickle.load(handle)
    dict_resultable_2_Y[run] = {'y_obs': dict_hp['y_obs'],
                              'n_l & n_n': [dict_hp['y_layers'], dict_hp['y_nodes']],  ' r2_Yf_train':
                             dict_predict_STATS_Y[run].loc[:, 'train_Yf'].mean(),' r2_Yf_valid':
                             dict_predict_STATS_Y[run].loc[:, 'valid_Yf'].mean(), ' r2_Yf_test':
                             dict_predict_STATS_Y[run].loc[:, 'test_Yf'].mean(),'lambda': dict_hp['regularization factor']}
df_resultable2_Y = pd.DataFrame(dict_resultable_2_Y).T.sort_values(by='y_obs')
print('============================================================================')
print('RESULT TABLE 2')
print(df_resultable2_Y)
print('============================================================================')




## RESULTS PLOT =========================================================================================================
## Plot for the K
Kx = dict_params['KxT_num'].T
E_complex = np.linalg.eigvals(Kx)

# K matrix heatmap
plt.figure(figsize=(20,20))
a = sb.heatmap(Kx, cmap="RdYlGn",center=0,vmax=np.abs(Kx).max(),vmin=-np.abs(Kx).max())
b, t = a.axes.get_ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
a.axes.set_ylim(b, t)
cbar = a.collections[0].colorbar
# here set the labelsize by 20
# cbar.ax.tick_params(labelsize=FONTSIZE)
# a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
# a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
plt.show()

## Eigenvalue plot
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot(1, 1, 1)
circ = plt.Circle((0, 0), radius=1, edgecolor='None', facecolor='cyan')
ax.add_patch(circ)
ax.plot(np.real(E_complex),np.imag(E_complex),'x',linewidth=3,color='g')
ax.set_xlabel('$Re(\lambda)$')
ax.set_ylabel('$Im(\lambda)$')
ax.set_xticks([-1.0,-0.5,0,0.5,1.0])
ax.set_yticks([-1.0,-0.5,0,0.5,1.0])
ax.set_xlim([-1.1,1.1])
ax.set_ylim([-1.1,1.1])
plt.show()

## Plotting the eigenfunctions

run = 9
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
Kx = dict_params['KxT_num'].T
e_in,W_in = np.linalg.eig(Kx)
E_in = np.diag(e_in)
E, W, comp_modes, comp_modes_conj = rnaf.resolve_complex_right_eigenvalues(E_in, W_in)
Winv = np.linalg.inv(W)

index = 0
n_funcs = 10
XT = np.concatenate([dict_ALLDATA['unscaled']['MX'][index]['XpT'][0:1,:],dict_ALLDATA['unscaled']['MX'][index]['XfT']],axis=0)
XTs = dict_ALLDATA['X_scaler'].transform(XT)
psiXTs_true = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: XTs})
psiXTs = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_ALLDATA['scaled']['MX'][index]['XpT'][0:1,:]})
for i in range(len(dict_ALLDATA['unscaled']['MX'][index]['XfT'])):
    psiXTs = np.concatenate([psiXTs,np.matmul(psiXTs[-1:],dict_params['KxT_num'])])
Phis = np.matmul(Winv, psiXTs.T)
# Phis = np.matmul(Winv, psiXTs_true.T)
YT = np.concatenate([dict_ALLDATA['unscaled']['MX'][index]['YpT'][0:1,:],dict_ALLDATA['unscaled']['MX'][index]['YfT']],axis=0)
YTs = dict_ALLDATA['Y_scaler'].transform(YT)

x_ticks = np.array([1,2,3,4,5,6,7])
f,ax = plt.subplots(5,2,sharex=True,figsize=(10,n_funcs*1.5))
ax = ax.reshape(-1)
eig_func_index = 0
for i in range(n_funcs):
    if eig_func_index in comp_modes:
        ax[i].plot(x_ticks, Phis[i, :],label = 'Real')
        ax[i].plot(x_ticks, Phis[i+1, :],label = 'Imaginary')
        # ax[i].legend()
        real_part = round(np.abs(E[eig_func_index,eig_func_index]),3)
        imag_part = round(np.abs(E[eig_func_index,eig_func_index+1]),3)
        ax[i].set_title('$\phi_{{{},{}}}(x)$'.format(eig_func_index+1,eig_func_index+2) + ', $\lambda =$' + str(real_part) +'$\pm$j' +  str(imag_part))
        eig_func_index = eig_func_index + 2
    else:
        ax[i].plot(x_ticks, Phis[i, :])
        real_part = round(np.abs(E[eig_func_index, eig_func_index]), 3)
        # print(real_part)
        # ax[i].set_title('$\phi_{{{}}}(x), \lambda = {{{}}}$'.format(eig_func_index+1,real_part))
        ax[i].set_title('$\phi_{{{}}}(x)$'.format(eig_func_index + 1) + ', $\lambda = $'+ str(real_part))
        eig_func_index = eig_func_index + 1
f.show()

ls_gene_max_var_index = sorted(range(len(XT.var(axis=0))), key=lambda i: XT.var(axis=0)[i])
ls_gene_max_var_index.reverse()
plt.figure(figsize=(10,6))
for i in range(9):
    plt.plot(x_ticks,XTs[:,ls_gene_max_var_index[i]],label = 'gene_' + str(i))
plt.legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
plt.show()


print(r2_score(psiXTs_true,psiXTs,multioutput='raw_values'))

# dict_x_index ={'MX': np.array([2,3,4,5,6,7]),'MN': np.array([4,5,6,7]),'NC': np.array([4,5,6,7])}
# for COND_NO in range(len(ALL_CONDITIONS)):
#     COND = ALL_CONDITIONS[COND_NO]
#     Xs = dict_ALLDATA['X_scaler'].transform(X)
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


tf.reset_default_graph()
sess.close()

##
run = 31

sess = tf.InteractiveSession()
run_folder_name = root_run_file + '/Sequential/RUN_' + str(run)
saver = tf.compat.v1.train.import_meta_graph(
    run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
dict_params = {}
dict_params['psixfT'] = tf.get_collection('psixfT')[0]
dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
dict_params['WhT_num'] = sess.run(tf.get_collection('WhT')[0])


# Wh matrix heatmap
Wh = dict_params['WhT_num'].T
plt.figure(figsize=(20,20))
a = sb.heatmap(Wh, cmap="RdYlGn",center=0,vmax=np.abs(Wh).max(),vmin=-np.abs(Wh).max())
b, t = a.axes.get_ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
a.axes.set_ylim(b, t)
cbar = a.collections[0].colorbar
# here set the labelsize by 20
# cbar.ax.tick_params(labelsize=FONTSIZE)
# a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
# a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
plt.show()


B = np.matmul(Wh,W)
plt.figure(figsize=(20,20))
a = sb.heatmap(B, cmap="RdYlGn",center=0,vmax=np.abs(B).max(),vmin=-np.abs(B).max())
b, t = a.axes.get_ylim()  # discover the values for bottom and top
b += 0.5  # Add 0.5 to the bottom
t -= 0.5  # Subtract 0.5 from the top
a.axes.set_ylim(b, t)
cbar = a.collections[0].colorbar
# here set the labelsize by 20
# cbar.ax.tick_params(labelsize=FONTSIZE)
# a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
# a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
plt.show()

Bmean = B.mean(axis=0)
ls_modes_ordered_by_output = sorted(range(len(np.abs(Bmean))), key=lambda i: np.abs(Bmean)[i])
ls_modes_ordered_by_output.reverse()

x_ticks = np.array([1,2,3,4,5,6,7])
f,ax = plt.subplots(5,2,sharex=True,figsize=(10,n_funcs*1.5))
ax = ax.reshape(-1)
eig_func_index = 0
for i in range(n_funcs):
    # if eig_func_index in comp_modes:
    #     ax[i].plot(x_ticks, Phis[i, :],label = 'Real')
    #     ax[i].plot(x_ticks, Phis[i+1, :],label = 'Imaginary')
    #     # ax[i].legend()
    #     real_part = round(np.abs(E[eig_func_index,eig_func_index]),3)
    #     imag_part = round(np.abs(E[eig_func_index,eig_func_index+1]),3)
    #     ax[i].set_title('$\phi_{{{},{}}}(x)$'.format(eig_func_index+1,eig_func_index+2) + ', $\lambda =$' + str(real_part) +'$\pm$j' +  str(imag_part))
    #     eig_func_index = eig_func_index + 2
    # else:
    ax[i].plot(x_ticks, Phis[ls_modes_ordered_by_output[i], :])
    real_part = round(np.abs(E[ls_modes_ordered_by_output[i], ls_modes_ordered_by_output[i]]), 3)
    # print(real_part)
    # ax[i].set_title('$\phi_{{{}}}(x), \lambda = {{{}}}$'.format(eig_func_index+1,real_part))
    ax[i].set_title('$\phi_{{{}}}(x)$'.format(eig_func_index + 1) + ', $\lambda = $'+ str(real_part))
    eig_func_index = eig_func_index + 1
f.show()

YT = np.concatenate([dict_ALLDATA['unscaled']['MX'][index]['YpT'][0:1,:],dict_ALLDATA['unscaled']['MX'][index]['YfT']],axis=0)
YTs = dict_ALLDATA['Y_scaler'].transform(YT)
plt.figure()
plt.plot(x_ticks,YTs)
plt.show()



tf.reset_default_graph()
sess.close()