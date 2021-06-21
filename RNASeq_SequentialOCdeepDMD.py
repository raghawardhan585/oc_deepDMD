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

SYSTEM_NO = 703 #410,411,412
ALL_CONDITIONS = ['MX']
ls_runs1 = list(range(0,60)) # SYSTEM 408

df_results2_deepDMD = rnaf.generate_n_step_prediction_table(SYSTEM_NO,ALL_CONDITIONS=['MX'],ls_runs1=list(range(0,100)),METHOD = 'Sequential')
ls_obs_deepDMD = list(df_results2_deepDMD.loc[:,'x_obs'].unique())
# ls_obs=[0,1,2,3,4]
dict_result3 = {}
for i in ls_obs_deepDMD:
    df_temp = copy.deepcopy(df_results2_deepDMD[df_results2_deepDMD['x_obs'] ==i])
    dict_result3[i] = {}
    dict_result3[i]['r2_X_nstep_train_mean'] = np.maximum(0,df_temp.loc[:,'r2_X_nstep_train']).mean()
    dict_result3[i]['r2_X_nstep_train_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_train']).std()
    dict_result3[i]['r2_X_nstep_valid_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).mean()
    dict_result3[i]['r2_X_nstep_valid_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_valid']).std()
    dict_result3[i]['r2_X_nstep_test_mean'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).mean()
    dict_result3[i]['r2_X_nstep_test_std'] = np.maximum(0,df_temp.loc[:, 'r2_X_nstep_test']).std()
df_results3_deepDMD = copy.deepcopy(pd.DataFrame(dict_result3).T)


## Save the best run of optimization problem 1
rnaf.save_best_run_of_Seq_OCdeepDMD_problem_1(SYSTEM_NO=703,RUN_NO=13)

## Performance of the output

## Output predictions

# SYSTEM_NO = 402
# ALL_CONDITIONS = ['MX']
# ls_runs1 = list(range(28,36)) # SYSTEM 402

SYSTEM_NO = 703
ALL_CONDITIONS = ['MX']
ls_runs1 = list(range(8,16)) # SYSTEM 402

# Generate predictions for each curve and write down the error statistics for each run
METHOD = 'Sequential'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
ls_all_run_indices = []
for folder in os.listdir(root_run_file + '/' + METHOD):
    if folder[0:4] == 'RUN_':  # It is a RUN folder
        ls_all_run_indices.append(int(folder[4:]))
ls_runs1 = set(ls_runs1).intersection(set(ls_all_run_indices))

# Indices [train, validation and test]
with open(indices_path, 'rb') as handle:
    ls_data_indices = pickle.load(handle)
ls_train_indices = ls_data_indices[0:12]
ls_valid_indices = ls_data_indices[12:14]
ls_test_indices = ls_data_indices[14:16]
dict_ALLDATA = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS=ALL_CONDITIONS)

dict_empty_all_conditions = {}
for COND in ALL_CONDITIONS:
    dict_empty_all_conditions[COND] = {}

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