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


# Preprocessing files
SYSTEM_NO = 400
# ALL_CONDITIONS = ['MX']
ALL_CONDITIONS = ['MX','MN']

ls_runs1 = list(range(4,60))
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

# Xp1=[]
# Xf1=[]
# for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices[0:14]):
#     try:
#         Xp1 = np.concatenate([Xp1, np.array(dict_data_original[COND][i]['df_X_TPM'])[:,0:-1].T],axis=0)
#         Xf1 = np.concatenate([Xf1, np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T], axis=0)
#     except:
#         Xp1 = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 0:-1].T
#         Xf1 = np.array(dict_data_original[COND][i]['df_X_TPM'])[:, 1:].T
#     print('cond ',COND,' i: ', i, ' shape:', len(Xp1))
#
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
    dict_params['psixfT'] = tf.get_collection('psixfT')[0]
    dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
    dict_params['WhT_num'] = sess.run(tf.get_collection('WhT')[0])

    # print('Run :', run, '  r2_train :', r2_score(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: Xf_train}), np.matmul(dict_params['psixpT'].eval(feed_dict ={dict_params['xpT_feed']: Xp_train}),dict_params['KxT_num']), multioutput='variance_weighted'))
    # print('         r2_valid :',
    #       r2_score(dict_params['psixfT'].eval(feed_dict={dict_params['xfT_feed']: Xf_valid}),
    #                np.matmul(dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: Xp_valid}),
    #                          dict_params['KxT_num']), multioutput='variance_weighted'))
    dict_instant_run_result = copy.deepcopy(dict_empty_all_conditions)
    for items in dict_instant_run_result.keys():
        dict_instant_run_result[items] = {'train_Yf_1step': [], 'valid_Yf_1step': [], 'test_Yf_1step': []}
    for COND,data_index in itertools.product(ALL_CONDITIONS, ls_data_indices):
        # Figure out if the index belongs to train, test or validation
        if data_index in ls_train_indices:
            key2_start = 'train_'
        elif data_index in ls_valid_indices:
            key2_start = 'valid_'
        else:
            key2_start = 'test_'
        # --- *** Generate prediction *** ---
        # Yf - 1 step
        YfT_hat = oc.inverse_transform_Y(np.matmul(dict_params['psixfT'].eval(feed_dict ={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), dict_params['WhT_num']),SYSTEM_NO)
        dict_instant_run_result[COND][key2_start + 'Yf_1step'].append(
            r2_score(dict_unscaled_data[COND][data_index]['YfT'].reshape(-1), YfT_hat.reshape(-1)))
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
          ['train_Yf_1step', 'valid_Yf_1step', 'test_Yf_1step']])
    print('=====================================================================')