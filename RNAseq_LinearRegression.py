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

def resolve_complex_right_eigenvalues(E, W):
    eval = copy.deepcopy(np.diag(E))
    comp_modes = []
    comp_modes_conj = []
    for i1 in range(E.shape[0]):
        if np.imag(E[i1, i1]) != 0:
            print(i1)
            # Find the complex conjugate
            for i2 in range(i1 + 1, E.shape[0]):
                if eval[i2] == eval[i1].conj():
                    break
            # i1 and i2 are the indices of the complex conjugate eigenvalues
            comp_modes.append(i1)
            comp_modes_conj.append(i2)
            E[i1, i1] = np.real(eval[i1])
            E[i2, i2] = np.real(eval[i1])
            E[i1, i2] = np.imag(eval[i1])
            E[i2, i1] = - np.imag(eval[i1])
            u1 = copy.deepcopy(np.real(W[:, i1:i1 + 1]))
            w1 = copy.deepcopy(np.imag(W[:, i1:i1 + 1]))
            W[:, i1:i1 + 1] = u1
            W[:, i2:i2 + 1] = w1
    E_out = np.real(E)
    W_out = np.real(W)
    return E_out, W_out, comp_modes, comp_modes_conj

# ## Importing all necessary information
SYSTEM_NO = 406
ALL_CONDITIONS = ['MX']
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'

# Scaler import
with open(root_run_file + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle','rb') as handle:
    All_Scalers = pickle.load(handle)
X_scaler = All_Scalers['X Scale']
Y_scaler = All_Scalers['Y Scale']


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
         'Xf': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
         'Yp': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
         'Yf': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}, SYSTEM_NO)
    dict_scaled_data[COND][i] = {'XpT': dict_intermediate['Xp'], 'XfT': dict_intermediate['Xf'],
                                 'YpT': dict_intermediate['Yp'], 'YfT': dict_intermediate['Yf']}
    dict_unscaled_data[COND][i] = {'XpT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                   'XfT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
                                   'YpT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
                                   'YfT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}


XpTs_train = XfTs_train = XpTs_valid = XfTs_valid = XpTs_test = XfTs_test = []
YpTs_train = YfTs_train = YpTs_valid = YfTs_valid = YpTs_test = YfTs_test = []
for COND,i in itertools.product(ALL_CONDITIONS,ls_train_indices):
    try:
        XpTs_train = np.concatenate([XpTs_train, dict_scaled_data[COND][i]['XpT']], axis=0)
        XfTs_train = np.concatenate([XfTs_train, dict_scaled_data[COND][i]['XfT']], axis=0)
        YpTs_train = np.concatenate([YpTs_train, dict_scaled_data[COND][i]['YpT']], axis=0)
        YfTs_train = np.concatenate([YfTs_train, dict_scaled_data[COND][i]['YfT']], axis=0)
    except:
        XpTs_train = dict_scaled_data[COND][i]['XpT']
        XfTs_train = dict_scaled_data[COND][i]['XfT']
        YpTs_train = dict_scaled_data[COND][i]['YpT']
        YfTs_train = dict_scaled_data[COND][i]['YfT']

for COND,i in itertools.product(ALL_CONDITIONS,ls_valid_indices):
    try:
        XpTs_valid = np.concatenate([XpTs_valid, dict_scaled_data[COND][i]['XpT']], axis=0)
        XfTs_valid = np.concatenate([XfTs_valid, dict_scaled_data[COND][i]['XfT']], axis=0)
        YpTs_valid = np.concatenate([XpTs_valid, dict_scaled_data[COND][i]['YpT']], axis=0)
        YfTs_valid = np.concatenate([XfTs_valid, dict_scaled_data[COND][i]['YfT']], axis=0)
    except:
        XpTs_valid = dict_scaled_data[COND][i]['XpT']
        XfTs_valid = dict_scaled_data[COND][i]['XfT']
        YpTs_valid = dict_scaled_data[COND][i]['YpT']
        YfTs_valid = dict_scaled_data[COND][i]['YfT']
XpTs_train_valid = np.concatenate([XpTs_train, XpTs_valid],axis=0)
XfTs_train_valid = np.concatenate([XfTs_train, XfTs_valid],axis=0)
YpTs_train_valid = np.concatenate([YpTs_train, YpTs_valid],axis=0)
YfTs_train_valid = np.concatenate([YfTs_train, YfTs_valid],axis=0)

for COND,i in itertools.product(ALL_CONDITIONS,ls_test_indices):
    try:
        XpTs_test = np.concatenate([XpTs_test, dict_scaled_data[COND][i]['XpT']], axis=0)
        XfTs_test = np.concatenate([XfTs_test, dict_scaled_data[COND][i]['XfT']], axis=0)
        YpTs_test = np.concatenate([YpTs_test, dict_scaled_data[COND][i]['YpT']], axis=0)
        YfTs_test = np.concatenate([YfTs_test, dict_scaled_data[COND][i]['YfT']], axis=0)
    except:
        XpTs_test = dict_scaled_data[COND][i]['XpT']
        XfTs_test = dict_scaled_data[COND][i]['XfT']
        YpTs_test = dict_scaled_data[COND][i]['YpT']
        YfTs_test = dict_scaled_data[COND][i]['YfT']

## Fitting state model

model_X = LinearRegression(fit_intercept=True) # Because of the scaling theorem, we always need a bias
model_X.fit(XpTs_train, XfTs_train)


# Properties of the eigenvalues
K1_without_intercept = np.concatenate([np.array(model_X.coef_),np.zeros(shape=(1,len(model_X.coef_)))],axis=0)
K1_intercept = np.concatenate([model_X.intercept_.reshape(-1,1),np.array([[1]])],axis=0)
K1 = np.concatenate([K1_without_intercept,K1_intercept], axis=1)
E_complex = np.linalg.eigvals(K1)
plt.figure()
plt.plot(1 - (np.cumsum(np.abs(E_complex)**2)/np.sum(np.abs(E_complex)**2)))
plt.xlabel('Number of $\lambda$s ($n$)')
plt.ylabel('$(\sum_{i=1}^n|\lambda_i|^2)/(\sum_{i=1}^N|\lambda_i|^2)$')
plt.title('Number of genes = '+ str(n_genes))
plt.xlim([0,100])
plt.xticks(list(range(0,100,10)))
plt.show()

# Predict on all the datasets
dict_results = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    # Predict the Xf - 1 step
    XfTs_hat = model_X.predict(dict_scaled_data[COND][i]['XpT'])
    # Predict the Xf - n step
    XfTsn_hat = dict_scaled_data[COND][i]['XpT'][0:1,:]
    for j in range(len(XfTs_hat)):
        XfTsn_hat = np.concatenate([XfTsn_hat,model_X.predict(XfTsn_hat[-1:,:])],axis=0)
    XfTsn_hat = XfTsn_hat[1:]
    # Reverse the Xfs
    XfT_hat = X_scaler.inverse_transform(XfTs_hat)
    XfTn_hat = X_scaler.inverse_transform(XfTsn_hat)
    # Compute and the r2 score
    dict_results[COND][i] = {}
    dict_results[COND][i]['r2_Xfs_1step'] = r2_score(dict_scaled_data[COND][i]['XfT'], XfTs_hat)#,multioutput ='variance_weighted')
    dict_results[COND][i]['r2_Xfs_nstep'] = r2_score(dict_scaled_data[COND][i]['XfT'], XfTsn_hat)#,multioutput ='variance_weighted')
    dict_results[COND][i]['r2_Xf_1step'] = r2_score(dict_unscaled_data[COND][i]['XfT'], XfT_hat)#,multioutput ='variance_weighted')
    dict_results[COND][i]['r2_Xf_nstep'] = r2_score(dict_unscaled_data[COND][i]['XfT'], XfTn_hat)#,multioutput ='variance_weighted')

df_results1 = pd.DataFrame(dict_results['MX'])
print(df_results1)

dict_results2 ={}
for cond in ALL_CONDITIONS:
    dict_results2[cond] ={}
    dict_results2[cond]['train_Xf_1step'] = df_results1.loc['r2_Xf_1step',ls_train_indices].mean()
    dict_results2[cond]['train_Xf_nstep'] = df_results1.loc['r2_Xf_nstep', ls_train_indices].mean()
    dict_results2[cond]['valid_Xf_1step'] = df_results1.loc['r2_Xf_1step', ls_valid_indices].mean()
    dict_results2[cond]['valid_Xf_nstep'] = df_results1.loc['r2_Xf_nstep', ls_valid_indices].mean()
    dict_results2[cond]['test_Xf_1step'] = df_results1.loc['r2_Xf_1step', ls_test_indices].mean()
    dict_results2[cond]['test_Xf_nstep'] = df_results1.loc['r2_Xf_nstep', ls_test_indices].mean()

df_results2 = pd.DataFrame(dict_results2).T
print(df_results2)

## Find the optimal linear output regression model

model_Y = LinearRegression(fit_intercept=True)
model_Y.fit(XfTs_train,YfTs_train)


# Predict on all the datasets
dict_resultsY = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    # Predict the Y - 1 step
    YfTs_hat = model_Y.predict(dict_scaled_data[COND][i]['XfT'])
    # Reverse the Ys
    YfT_hat = Y_scaler.inverse_transform(YfTs_hat)
    # Compute and the r2 score
    dict_resultsY[COND][i] = {}
    dict_resultsY[COND][i]['r2_Yfs'] = r2_score(dict_scaled_data[COND][i]['YfT'].reshape(-1), YfTs_hat.reshape(-1))
    dict_resultsY[COND][i]['r2_Yf'] = r2_score(dict_unscaled_data[COND][i]['YfT'].reshape(-1), YfT_hat.reshape(-1))

dict_df_resultsY = {}
for COND in ALL_CONDITIONS:
    dict_df_resultsY[COND] = pd.DataFrame(dict_resultsY[COND]).T.loc[:,['r2_Yfs','r2_Yf']]
    print(dict_df_resultsY[COND])
dict_mean_resultsY = copy.deepcopy(dict_empty_all_conditions)
for COND in ALL_CONDITIONS:
    dict_mean_resultsY[COND]['train_Yf'] = dict_df_resultsY[COND].loc[ls_train_indices,'r2_Yf'].mean()
    dict_mean_resultsY[COND]['valid_Yf'] = dict_df_resultsY[COND].loc[ls_valid_indices, 'r2_Yf'].mean()
    dict_mean_resultsY[COND]['test_Yf'] = dict_df_resultsY[COND].loc[ls_test_indices, 'r2_Yf'].mean()

print('-------------------------')
df_mean_resultsY = pd.DataFrame(dict_mean_resultsY).T.loc[:,['train_Yf','valid_Yf','test_Yf']]
print(df_mean_resultsY)

U,S,V = np.linalg.svd(model_Y.coef_)
plt.figure()
plt.plot(1 - (np.cumsum(S**2)/np.sum(S**2)))
plt.xlabel('Number of $\lambda$s ($n$)')
plt.ylabel('$(\sum_{i=1}^n|s_i|^2)/(\sum_{i=1}^N|s_i|^2)$')
plt.title('Number of genes = '+ str(n_genes))
plt.xlim([0,20])
plt.xticks(list(range(0,20,2)))
plt.show()