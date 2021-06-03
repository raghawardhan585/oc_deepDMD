import pickle
# import random
import numpy as np
import pandas as pd
import os
# import shutil
# import random
import matplotlib.pyplot as plt
import copy
import itertools
# import seaborn as sb
# import tensorflow as tf

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import numpy as np
from sklearn.metrics import make_scorer,r2_score
pd.set_option("display.max_rows", None, "display.max_columns", None)
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

# Preprocessing files
SYSTEM_NO = 401
ALL_CONDITIONS = ['MX']
# SYSTEM_NO = 400
# ALL_CONDITIONS = ['MX','MN']

# original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
# indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_file = 'System_' + str(SYSTEM_NO)
# Indices [train and test]
with open(root_file + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle','rb') as handle:
    ls_data_indices = pickle.load(handle)
ls_train_indices = ls_data_indices[0:12]
ls_valid_indices = ls_data_indices[12:14]
ls_train_valid_indices = ls_data_indices[0:14]
ls_test_indices = ls_data_indices[14:16]
# Scaler import
with open(root_file + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle','rb') as handle:
    All_Scalers = pickle.load(handle)
X_scaler = All_Scalers['X Scale']
Y_scaler = All_Scalers['Y Scale']
# Datasets [sorted as scaled and unscaled] and Conditions
with open(root_file + '/System_' + str(SYSTEM_NO) + '_Data.pickle','rb') as handle:
    dict_data_original = pickle.load(handle)
n_genes = len(dict_data_original[ALL_CONDITIONS[0]][ls_data_indices[0]]['df_X_TPM'])
dict_empty_all_conditions ={}
for COND in ALL_CONDITIONS:
    dict_empty_all_conditions[COND] = {}
dict_scaled_data = copy.deepcopy(dict_empty_all_conditions)
dict_unscaled_data = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    dict_unscaled_data[COND][i] = {'XpT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                   'XfT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
                                   'YpT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
                                   'YfT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}
    dict_scaled_data[COND][i] = {'XpT': X_scaler.transform(dict_unscaled_data[COND][i]['XpT']),
                                   'XfT': X_scaler.transform(dict_unscaled_data[COND][i]['XfT']),
                                   'YpT': Y_scaler.transform(dict_unscaled_data[COND][i]['YpT']),
                                   'YfT': Y_scaler.transform(dict_unscaled_data[COND][i]['YfT'])}

## --------------------------------------------------------------------------------
# DMD Train with Lasso Regression and k-fold cross validation

# Notes - This does not require a validation data. We use each one of the k-folds as the validation to draw a statistic
# on which is the most robust hyperparameter (\lambda - the lasso regularization parameter)
XpTs_train = XfTs_train = XpTs_valid = XfTs_valid = XpTs_test = XfTs_test = []
for COND,i in itertools.product(ALL_CONDITIONS,ls_train_indices):
    try:
        XpTs_train = np.concatenate([XpTs_train, dict_scaled_data[COND][i]['XpT']],axis=0)
        XfTs_train = np.concatenate([XfTs_train, dict_scaled_data[COND][i]['XfT']],axis=0)
    except:
        XpTs_train = dict_scaled_data[COND][i]['XpT']
        XfTs_train = dict_scaled_data[COND][i]['XfT']
for COND,i in itertools.product(ALL_CONDITIONS,ls_valid_indices):
    try:
        XpTs_valid = np.concatenate([XpTs_valid, dict_scaled_data[COND][i]['XpT']],axis=0)
        XfTs_valid = np.concatenate([XfTs_valid, dict_scaled_data[COND][i]['XfT']],axis=0)
    except:
        XpTs_valid = dict_scaled_data[COND][i]['XpT']
        XfTs_valid = dict_scaled_data[COND][i]['XfT']
XpTs_train_valid = np.concatenate([XpTs_train, XpTs_valid],axis=0)
XfTs_train_valid = np.concatenate([XfTs_train, XfTs_valid],axis=0)
for COND,i in itertools.product(ALL_CONDITIONS,ls_test_indices):
    try:
        XpTs_test = np.concatenate([XpTs_test, dict_scaled_data[COND][i]['XpT']],axis=0)
        XfTs_test = np.concatenate([XfTs_test, dict_scaled_data[COND][i]['XfT']],axis=0)
    except:
        XpTs_test =dict_scaled_data[COND][i]['XpT']
        XfTs_test = dict_scaled_data[COND][i]['XfT']



# Hyperparameters of choice
# Lasso_reg_lambda = np.arange(1e-3,11e-3,1e-3)
Lasso_reg_lambda = [0,2]
# Lasso_reg_lambda = np.arange(0,1.1,0.1)
# Lasso_reg_lambda = np.concatenate([Lasso_reg_lambda, np.arange(0.02,0.1,0.005)])
# Lasso_reg_lambda = np.sort(Lasso_reg_lambda)

NO_OF_FOLDS = 7
kf = KFold(n_splits=NO_OF_FOLDS, shuffle=False, random_state=None)
# my_scorer = make_scorer(r2_score,multioutput='uniform_average')
my_scorer = make_scorer(r2_score,multioutput='variance_weighted')

# [Note]: kf.split takes the indices of the input array and then then the kfold split of the indices
# [Note- fit_intercept=True]: Our theorem states that since we do an affine transformation while scaling, we should always have a bias term and hence we fit the intercept

dict_stats = {}
for alpha in Lasso_reg_lambda:
    dict_stats[alpha] = {}
    if alpha ==0:
        a =cross_val_score(LinearRegression(fit_intercept=True), XpTs_train_valid, XfTs_train_valid, cv=kf.split(XpTs_train_valid),scoring=my_scorer)
    else:
        if alpha<0.1:
            a = cross_val_score(Lasso(alpha= alpha, fit_intercept=True, max_iter=50000), XpTs_train_valid, XfTs_train_valid, cv=kf.split(XpTs_train_valid), scoring=my_scorer)
        else:
            a = cross_val_score(Lasso(alpha=alpha, fit_intercept=True, max_iter=5000), XpTs_train_valid, XfTs_train_valid, cv=kf.split(XpTs_train_valid), scoring=my_scorer)
    for i in range(NO_OF_FOLDS):
        dict_stats[alpha][i] = a[i]
    print('[STATUS COMPLETE] Lambda = ',alpha)
    print(a)

df_stats = pd.DataFrame(dict_stats).T
print(df_stats)


## Saving results of lasso regression
file_save_path = root_file + '/Lasso_Regression_X'
# Make tha save path if a  path does not exist
if not os.path.isdir(file_save_path):
    os.mkdir(file_save_path)
# Appending and saving the result
try:
    # Scheme 2 : append the results to an existing pickle file [only use if you have the same number of folds (kfold cross validation)]
    # Opening the old lasso regression results
    with open(file_save_path + '/Lasso_stats.pickle','rb') as handle:
        df_stats1 = pickle.load(handle)
    if df_stats1.columns == df_stats.columns:
        # Concatenating to the new ones
        df_stats_new = pd.concat([df_stats1,df_stats],axis=0).T.sort_index(axis=1).T
        # Saving the concatenated lasso regression results
        with open(file_save_path + '/Lasso_stats.pickle','wb') as handle:
            pickle.dump(df_stats_new, handle)
        print(df_stats_new)
    else:
        print('The column indices do not match between the two dataframes !!!')
        print('Saving the results as a proxy')
        print('PLEASE RESOLVE IT !!!')
        with open(file_save_path + '/Lasso_stats_proxy.pickle', 'wb') as handle:
            pickle.dump(df_stats, handle)
        df_stats_new = df_stats
except:
    # Scheme 1 : save the results to a new pickle
    with open(file_save_path + '/Lasso_stats.pickle', 'wb') as handle:
        pickle.dump(df_stats, handle)
    df_stats_new = df_stats

print('====================================================================================')
print(' ALL COMBINED STATS:')
print(df_stats_new)
print('====================================================================================')
## Find the optimal Lasso regression model

dict_optimalLasso = {}
# Finding the optimal lambda corresponding to max mean r^2 value closest to 1
mean_r2 = df_stats_new.mean(axis=1)
dict_optimalLasso['lambda'] = mean_r2[mean_r2 == mean_r2.max()].index[0]

# Fit the optimal model on all the training data
if dict_optimalLasso['lambda'] ==0:
    print('[NOTE] Optimal model was without Lasso regularization')
    opt_model = LinearRegression(fit_intercept=True)
else:
    opt_model = Lasso(alpha = dict_optimalLasso['lambda'], fit_intercept=True, max_iter=50000)

opt_model.fit(XpTs_train,XfTs_train)
dict_optimalLasso['model'] = opt_model

# Predict on all the datasets
dict_results = copy.deepcopy(dict_empty_all_conditions)
for COND,i in itertools.product(ALL_CONDITIONS,ls_data_indices):
    # Predict the Xf - 1 step
    XfTs_hat = opt_model.predict(dict_scaled_data[COND][i]['XpT'])
    # Predict the Xf - n step
    XfTsn_hat = dict_scaled_data[COND][i]['XpT'][0:1,:]
    for j in range(len(XfTs_hat)):
        XfTsn_hat = np.concatenate([XfTsn_hat,opt_model.predict(XfTsn_hat[-1:,:])],axis=0)
    XfTsn_hat = XfTsn_hat[1:]
    # Reverse the Xfs
    XfT_hat = X_scaler.inverse_transform(XfTs_hat)
    XfTn_hat = X_scaler.inverse_transform(XfTsn_hat)
    # Compute and the r2 score
    dict_results[COND][i] = {}
    dict_results[COND][i]['r2_Xfs_1step'] = r2_score(dict_scaled_data[COND][i]['XfT'], XfTs_hat,multioutput ='variance_weighted')
    dict_results[COND][i]['r2_Xfs_nstep'] = r2_score(dict_scaled_data[COND][i]['XfT'], XfTsn_hat,multioutput ='variance_weighted')
    dict_results[COND][i]['r2_Xf_1step'] = r2_score(dict_unscaled_data[COND][i]['XfT'], XfT_hat,multioutput ='variance_weighted')
    dict_results[COND][i]['r2_Xf_nstep'] = r2_score(dict_unscaled_data[COND][i]['XfT'], XfTn_hat,multioutput ='variance_weighted')


dict_df_results = {}
for COND in ALL_CONDITIONS:
    dict_df_results[COND] = pd.DataFrame(dict_results[COND]).T.loc[:,['r2_Xfs_1step','r2_Xfs_nstep','r2_Xf_1step','r2_Xf_nstep']]
    print(dict_df_results[COND])
dict_mean_results = copy.deepcopy(dict_empty_all_conditions)
for COND in ALL_CONDITIONS:
    dict_mean_results[COND]['train_X_1step'] = dict_df_results[COND].loc[ls_train_indices,'r2_Xf_1step'].mean()
    dict_mean_results[COND]['valid_X_1step'] = dict_df_results[COND].loc[ls_valid_indices, 'r2_Xf_1step'].mean()
    dict_mean_results[COND]['test_X_1step'] = dict_df_results[COND].loc[ls_test_indices, 'r2_Xf_1step'].mean()
    dict_mean_results[COND]['train_X_nstep'] = dict_df_results[COND].loc[ls_train_indices, 'r2_Xf_nstep'].mean()
    dict_mean_results[COND]['valid_X_nstep'] = dict_df_results[COND].loc[ls_valid_indices, 'r2_Xf_nstep'].mean()
    dict_mean_results[COND]['test_X_nstep'] = dict_df_results[COND].loc[ls_test_indices, 'r2_Xf_nstep'].mean()
print('-------------------------')
df_mean_results = pd.DataFrame(dict_mean_results).T.loc[:,['train_X_1step','valid_X_1step','test_X_1step','train_X_nstep','valid_X_nstep','test_X_nstep']]
print(df_mean_results)
dict_optimalLasso['stats'] = df_mean_results

# Saving the optimal model
with open(file_save_path + '/Lasso_optimal_model.pickle', 'wb') as handle:
    pickle.dump(dict_optimalLasso, handle)

