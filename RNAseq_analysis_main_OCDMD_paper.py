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

def ADD_BIAS_ROW(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = np.concatenate([X_IN, np.ones(shape=(1, X_IN.shape[1]))], axis=0)
    else:
        X_OUT = X_IN
    return X_OUT
def ADD_BIAS_COLUMN(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = np.concatenate([X_IN, np.ones(shape=(X_IN.shape[0], 1))], axis=1)
    else:
        X_OUT = X_IN
    return X_OUT
def REMOVE_BIAS_ROW(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = X_IN[0:-1,:]
    else:
        X_OUT = X_IN
    return X_OUT
def REMOVE_BIAS_COLUMN(X_IN,ADD_BIAS):
    if ADD_BIAS:
        X_OUT = X_IN[:,0:-1]
    else:
        X_OUT = X_IN
    return X_OUT
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

## Importing all necessary information
SYSTEM_NO = 304
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
# Import the training data
with open(ocdeepDMD_data_path,'rb') as handle:
    dict_DMD_train_unbiased = pickle.load(handle) # already scaled, it is in transposed format though not stated
# Since the data is scaled, we use the affine transformation theorem to impose a bias constraint on the model
dict_DMD_train = copy.deepcopy(dict_DMD_train_unbiased)
dict_DMD_train['Xp'] = ADD_BIAS_COLUMN(dict_DMD_train['Xp'],True)
dict_DMD_train['Xf'] = ADD_BIAS_COLUMN(dict_DMD_train['Xf'],True)
# Import the full dataset and the indices
with open(original_data_path,'rb') as handle:
    dict_data_original = pickle.load(handle)
# Import the ordered indices - Last two indices are used as testing data
with open(indices_path,'rb') as handle:
    ls_data_indices = pickle.load(handle)
ALL_CONDITIONS = list(dict_data_original.keys())
# Setup the prediction dictionary for all datasets
dict_DMD_results = {}
for COND in ALL_CONDITIONS:
    dict_DMD_results[COND] = {}
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    dict_DMD_results[COND][i] = {}
    dict_DMD_results[COND][i]['Xp'] = np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:,0:-1])
    dict_DMD_results[COND][i]['Xf'] = np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:])
    dict_DMD_results[COND][i]['Yp'] = np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1])
    dict_DMD_results[COND][i]['Yf'] = np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:])
# The scalers can be called directly using functions

# ## --------------------------------------------------------------------------------
# # DMD Train with Lasso Regression and k-fold cross validation
#
# # Notes - This does not require a validation data. We use each one of the k-folds as the validation to draw a statistic
# # on which is the most robust hyperparameter (\lambda - the lasso regularization parameter)
#
# NO_OF_FOLDS = 14
#
# kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
# my_scorer = make_scorer(r2_score,multioutput='uniform_average')
# # print(cross_val_score(LinearRegression(fit_intercept=False), dict_DMD_train['Xp'], dict_DMD_train['Xf'], cv=kf.split(dict_DMD_train['Xp']),scoring=my_scorer))
# # print(cross_val_score(Lasso(alpha= 0.02, fit_intercept=False, max_iter=50000), dict_DMD_train['Xp'], dict_DMD_train['Xf'],cv=kf.split(dict_DMD_train['Xp']), scoring=my_scorer))
#
# dict_stats = {}
# for alpha in np.arange(0.0,0.5,0.5):
#     dict_stats[alpha] = {}
#     if alpha ==0:
#         a =cross_val_score(LinearRegression(fit_intercept=False), dict_DMD_train['Xp'], dict_DMD_train['Xf'], cv=kf.split(dict_DMD_train['Xp']),scoring=my_scorer)
#     else:
#         a = cross_val_score(Lasso(alpha= alpha, fit_intercept=False, max_iter=50000), dict_DMD_train['Xp'], dict_DMD_train['Xf'],cv=kf.split(dict_DMD_train['Xp']), scoring=my_scorer)
#     for i in range(NO_OF_FOLDS):
#         dict_stats[alpha][i] = a[i]
#     print('[STATUS COMPLETE] alpha = ',alpha)
#     print(a)
#
# df_stats = pd.DataFrame(dict_stats)
# print(df_stats)


# ## Saving results of lasso regression
#
# try:
#     # Scheme 2 : append the results to an existing pickle file [only use if you have the same number of folds (kfold cross validation)]
#     # Opening the old lasso regression results
#     with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','rb') as handle:
#         df_stats1 = pickle.load(handle)
#     # Concatenating to the new ones
#     df_stats_new = pd.concat([df_stats1,df_stats],axis=1).sort_index(axis=1)
#     # Saving the concatenated lasso regression results
#     with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','wb') as handle:
#         pickle.dump(df_stats_new, handle)
# except:
#     # Scheme 1 : save the results to a new pickle
#     with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle', 'wb') as handle:
#         pickle.dump(df_stats, handle)

## How does the linear model fit on all the data

lin_model1_X = LinearRegression(fit_intercept=True).fit(dict_DMD_train_unbiased['Xp'],dict_DMD_train_unbiased['Xf'])
lin_model1_Y = LinearRegression(fit_intercept=True).fit(np.concatenate([dict_DMD_train_unbiased['Xf'],dict_DMD_train_unbiased['Xp']],axis=0),np.concatenate([dict_DMD_train_unbiased['Yf'],dict_DMD_train_unbiased['Yp']],axis=0))
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
plt.figure()
dict_start_time_Yf = {'MX':2, 'MN':4, 'NC':4}
DOWNSAMPLE_FACTOR = 4
ls_train_indices = list(dict_DMD_results[ALL_CONDITIONS[0]].keys())[0:-2]
ls_test_indices = list(dict_DMD_results[ALL_CONDITIONS[0]].keys())[-2:]
for COND_NO in range(len(ALL_CONDITIONS)):
    COND = ALL_CONDITIONS[COND_NO]
    x_time = np.ones(shape=(np.product(dict_DMD_results[COND][0]['Yf'].shape)))
    x_time[0] = 0
    x_time = dict_start_time_Yf[COND] + np.cumsum(np.ones(shape=(np.product(dict_DMD_results[COND][i]['Yf'].shape))) * 3 / 60)
    for items in dict_DMD_results[COND].keys():
        # Scale the Xp
        data_scaled = oc.scale_data_using_existing_scaler_folder({'Xp': dict_DMD_results[COND][items]['Xp'].T, 'Xf': dict_DMD_results[COND][items]['Xf'].T, 'Yp': dict_DMD_results[COND][items]['Yp'].T, 'Yf': dict_DMD_results[COND][items]['Yf'].T}, SYSTEM_NO)
        XpTs = data_scaled['Xp']
        XfTs = data_scaled['Xf']
        YpTs = data_scaled['Yp']
        YfTs = data_scaled['Yf']

        # Predict the Xf - 1 step
        XfTs_hat = lin_model1_X.predict(XpTs)
        YpTs_hat = lin_model1_Y.predict(XpTs)
        YfTs_hat = lin_model1_Y.predict(XfTs_hat)


        # Predict the Xf - n step
        XfTsn_hat = XpTs[0:1,:]
        for i in range(len(XfTs_hat)):
            XfTsn_hat = np.concatenate([XfTsn_hat,lin_model1_X.predict(XfTsn_hat[-1:])],axis=0)
        XfTsn_hat = XfTsn_hat[1:]
        YfTsn_hat = lin_model1_Y.predict(XfTsn_hat)

        # Reverse the Xfs
        Xf_hat = oc.inverse_transform_X(XfTs_hat, SYSTEM_NO).T
        Yp_hat = oc.inverse_transform_Y(YpTs_hat, SYSTEM_NO).T
        Yf_hat = oc.inverse_transform_Y(YfTs_hat, SYSTEM_NO).T

        Xfn_hat = oc.inverse_transform_X(XfTsn_hat, SYSTEM_NO).T
        Yfn_hat = oc.inverse_transform_Y(YfTsn_hat, SYSTEM_NO).T

        # if items in ls_train_indices:
        #     plt.plot(x_time[0:-1:DOWNSAMPLE_FACTOR],
        #              dict_DMD_results[COND][items]['Yf'].T.reshape(-1)[0:-1:DOWNSAMPLE_FACTOR], '.',
        #              color=ls_colors[COND_NO])
        #     plt.plot(x_time, Yfn_hat.T.reshape(-1), color = ls_colors[COND_NO])
        if items in ls_test_indices:
            plt.plot(x_time[0:-1:DOWNSAMPLE_FACTOR],
                     dict_DMD_results[COND][items]['Yf'].T.reshape(-1)[0:-1:DOWNSAMPLE_FACTOR], '.',
                     color=ls_colors[COND_NO])
            if items == ls_test_indices[-1]:
                plt.plot(x_time, Yfn_hat.T.reshape(-1), color=ls_colors[COND_NO], label = COND)
            else:
                plt.plot(x_time, Yfn_hat.T.reshape(-1), color = ls_colors[COND_NO])
        # Compute and the r2 score
        dict_DMD_results[COND][items]['r2_Xfs_1step'] = r2_score(XfTs, XfTs_hat,multioutput ='variance_weighted')
        dict_DMD_results[COND][items]['r2_Yps_1step'] = r2_score(YpTs.reshape(-1), YpTs_hat.reshape(-1))
        dict_DMD_results[COND][items]['r2_Yfs_1step'] = r2_score(YfTs.reshape(-1), YfTs_hat.reshape(-1))
        dict_DMD_results[COND][items]['r2_Xfs_nstep'] = r2_score(XfTs, XfTsn_hat,multioutput ='variance_weighted')
        dict_DMD_results[COND][items]['r2_Yfs_nstep'] = r2_score(YfTs.reshape(-1), YfTsn_hat.reshape(-1))

        dict_DMD_results[COND][items]['r2_Xf_1step'] = r2_score(dict_DMD_results[COND][items]['Xf'].T, Xf_hat.T,multioutput ='variance_weighted')
        dict_DMD_results[COND][items]['r2_Yp_1step'] = r2_score(dict_DMD_results[COND][items]['Yp'].reshape(-1), Yp_hat.reshape(-1))
        dict_DMD_results[COND][items]['r2_Yf_1step'] = r2_score(dict_DMD_results[COND][items]['Yf'].reshape(-1), Yf_hat.reshape(-1))
        dict_DMD_results[COND][items]['r2_Xf_nstep'] = r2_score(dict_DMD_results[COND][items]['Xf'].T, Xfn_hat.T,multioutput ='variance_weighted')
        dict_DMD_results[COND][items]['r2_Yf_nstep'] = r2_score(dict_DMD_results[COND][items]['Yf'].reshape(-1), Yfn_hat.reshape(-1))
# a0 = plt.plot([0, 2],color = ls_colors[0],linestyle ='solid',label='MAX',linewidth = 1)
# a1 = plt.plot([0,2],color = ls_colors[1],linestyle ='solid',label = 'MIN',linewidth = 1)
# a2 = plt.plot([0,2],color = ls_colors[2],linestyle ='solid',label = 'NC',linewidth = 1)
# l2 = plt.legend((a0,a1,a2),('MAX','MIN','NC'),loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
plt.legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
plt.xlabel('Time [hrs]')
plt.ylabel('Fitness Score')
# plt.title('Testing Data')
plt.show()
df_results ={}
for COND in ALL_CONDITIONS:
    df_results[COND] = pd.DataFrame(dict_DMD_results[COND]).T.loc[:,['r2_Xfs_1step','r2_Yps_1step','r2_Yfs_1step','r2_Yp_1step','r2_Yf_1step','r2_Xfs_nstep','r2_Yfs_nstep','r2_Xf_1step','r2_Xf_nstep','r2_Yf_nstep']]
    print(df_results[COND])

for COND in ALL_CONDITIONS:
    print('-------------------------')
    print(COND, ' CONDITION TEST DATA')
    print('-------------------------')
    # print(df_results[COND].iloc[-2:, :])
    print(df_results[COND].iloc[-2:, :].mean(axis=0))
# ##
# with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_LasssoRegression_Results.pickle','rb') as handle:
#     df_stats1 = pickle.load(handle)
# df_stats_filt = np.maximum(df_stats1, 0)
#
# for i in df_stats_filt.columns:
#     if i>0.5:
#         df_stats_filt = df_stats_filt.drop(columns=[i])
# for i in df_stats_filt.index:
#     if np.sum(df_stats_filt.loc[i,:])==0:
#         df_stats_filt = df_stats_filt.drop(index=[i])
# plt.plot(df_stats_filt.columns, df_stats_filt.T, '.',color= '#ff7f0e')
# # plt.plot(df_stats_filt.columns,df_stats_filt.median(axis=0),color = '#ff7f0e')
# plt.plot(df_stats_filt.columns,df_stats_filt.mean(axis=0), color = '#1f77b4',marker='.',markersize = 10,alpha = 1)
# plt.errorbar(df_stats_filt.columns,df_stats_filt.mean(axis=0),df_stats_filt.std(axis=0), color = '#1f77b4', capsize = 8,fmt='.',alpha = 0.8)
# plt.xlabel('Lasso hyperparameter ($\lambda$)')
# plt.ylabel('$r^2$')
# plt.yticks([0,0.3,0.6,0.9])
# plt.xticks([0,0.1,0.2,0.3,0.4])
# plt.xlim([-0.005,0.4])
# plt.ylim([-0.005,1])
# plt.show()
# ## Fit lasso regression models and
#
# model_lin_reg = LinearRegression(fit_intercept=False).fit()




## Modes of the system
num_modes = 195

# Construct the K matrix
K1_without_intercept = np.concatenate([np.array(lin_model1_X.coef_),np.zeros(shape=(1,len(lin_model1_X.coef_)))],axis=0)
K1_intercept = np.concatenate([lin_model1_X.intercept_.reshape(-1,1),np.array([[1]])],axis=0)
K1 = np.concatenate([K1_without_intercept,K1_intercept], axis=1)


# TODO Do an eigen decomposition and modal reduction
#  (keeping in mind that complex eigenvalues might be in the system)
E_complex,W_complex = np.linalg.eig(K1)
E, W, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(np.diag(E_complex),W_complex)
Winv = np.linalg.inv(W)

# Plot of the eigenvalues
plt.plot(np.abs(E_complex))
plt.xlabel('Index of $\lambda$')
plt.ylabel('$\lambda$')
# plt.xlim([0,250])
plt.show()
plt.plot((np.cumsum(np.abs(E_complex)**2)/np.sum(np.abs(E_complex)**2)))
plt.xlabel('Number of $\lambda$s ($n$)')
plt.ylabel('$(\sum_{i=1}^n|\lambda_i|^2)/(\sum_{i=1}^N|\lambda_i|^2)$')
plt.xlim([0,250])
plt.show()


## TODO Plot of the evolution of the eigenfunctions
n_funcs = 10
f,ax = plt.subplots(n_funcs,1,sharex=True,figsize=(7,n_funcs*1.5))
dict_x_index ={'MX': np.array([2,3,4,5,6,7]),'MN': np.array([4,5,6,7]),'NC': np.array([4,5,6,7])}
for COND_NO in range(len(ALL_CONDITIONS)):
    COND = ALL_CONDITIONS[COND_NO]
    Xps = oc.scale_data_using_existing_scaler_folder({'Xp':dict_DMD_results[COND][0]['Xp'].T},SYSTEM_NO)['Xp'].T
    Xps = np.concatenate([Xps, np.ones(shape=(1,len(Xps[0])))], axis = 0)
    Phis = np.matmul(Winv, Xps)
    Phis = Phis[0:-1,:]
    Phi = oc.inverse_transform_X(Phis.T,SYSTEM_NO).T
    for i in range(n_funcs):
        if i==0:
            ax[i].plot(dict_x_index[COND], Phi[i, :],label = COND)
        else:
            ax[i].plot(dict_x_index[COND], Phi[i,:])
        ax[i].set_title('$\lambda = $'+ str(E[i][i]))

ax[0].legend(loc = "lower center",bbox_to_anchor=(0.5,1.005),fontsize = 22,ncol=3)
ax[-1].set_xlabel('Time [hrs]')
f.show()
## TODO Plot of the modes reconstructing the profile of the dynamics
W_unscaled = oc.inverse_transform_X(W[0:-1].T, SYSTEM_NO).T
n_plots = 10
plt.figure()
plt.plot(W_unscaled[:,0:1])
plt.show()





##

