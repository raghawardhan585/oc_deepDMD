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
import itertools
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, make_scorer
import time
import scipy.stats as st
import operator
import seaborn as sns
from n_step_predictions import *
import copy
from scipy.signal import savgol_filter as sg_filter

plt.rcParams["font.family"] = "Avenir"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22
# plt.rcParams["transparent"] = True


##
N_CONCATENATED_OUTPUTS = 1
rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output=True, get_full_output=False, n_outputs=N_CONCATENATED_OUTPUTS, sf_filter_window_length = 15, sf_filter_polyorder = 3) # Getting the raw RNAseq and OD600 data to the state and output data format
# rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= False,n_outputs= 20, sf_filter_window_length = -1, sf_filter_polyorder = 3) # Getting the raw RNAseq and OD600 data to the state and output data format
# rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= True,n_outputs= -1)
with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)

REMOVE_NC_EQUILIBRIUM = True
SUBTRACT_FINAL_TIME_POINT = False
# SUBTRACT_FINAL_TIME_POINT = False
NORMALIZE_BY_NC_EQUILIBRIUM = False

# ls_time = np.arange(0,len(np.array(dict_DATA_ORIGINAL['MX'][1]['Y']).reshape(-1)))* 3/60
# plt.plot(ls_time,np.array(dict_DATA_ORIGINAL['MX'][1]['Y']).T.reshape(-1))
# plt.plot(ls_time[40:],np.array(dict_DATA_ORIGINAL['MN'][1]['Y']).T.reshape(-1))
# plt.plot(ls_time[40:],np.array(dict_DATA_ORIGINAL['NC'][1]['Y']).T.reshape(-1))
# plt.title('Output curves \n (NC Equilibrium subtracted)')
# plt.xlabel('Time (hrs)')
# plt.show()
##
def filter_genes_by_flatness_test(dict_data_in, significance_level_percent = 1e-3, ls_time_points = [1,2,3,4,5,6,7], n_lags =-1):
    # Parameters of the input
    ls_conditions = list(dict_data_in.keys())  # condition identifiers
    ls_replicates = list(dict_data_in[ls_conditions[0]].keys())  # replicate identifiers
    ls_replicates.sort()
    ngenes = dict_data_in[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].shape[0]  # number of genes
    ls_genes = list(dict_data_in[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].index)  # gene list

    # convert the given dictionary of dataframes to a single 3D matrix with each dataframe representing the dynamics of genes for a single initial conditions
    # for initial conditions
    n_timepts = len(ls_time_points)
    if n_lags ==-1:
        n_lags = np.int(np.ceil(n_timepts/2))

    np3Dall = np.empty(shape=(ngenes,n_timepts,0))
    for cond, rep in itertools.product(ls_conditions, ls_replicates):
        np3Dall = np.concatenate([np3Dall,np.array(pd.DataFrame(dict_data_in[cond][rep]['df_X_TPM'], columns=ls_time_points)).reshape(ngenes,-1,1)],axis=2)

    # for a model of the form y = c, we analyze the residuals using the autocorrelation function
    n_rejects = 0
    ls_filtered_genes =[]
    count =0
    for i in range(ngenes): # i indicates gene number
        data_i = np3Dall[i,:,:].T # rows are individual time traces, columns are time points for gene with gene number i
        n_pts_i = np.count_nonzero(~np.isnan(data_i)) # number of non nan data points
        mean_i = np.nanmean(data_i)
        data_i_mr = data_i - mean_i
        ls_autcov_i = []
        for l in range(n_lags+1):
            ls_autcov_i.append(np.nansum(data_i_mr[:,0:n_timepts - l]*data_i_mr[:,l:7]))
        ls_autocorr_i = ls_autcov_i/ls_autcov_i[0]
        UB = st.norm.ppf(1 - significance_level_percent/100/2)/np.sqrt(n_pts_i)
        LB = -UB
        if np.sum(np.logical_or((ls_autocorr_i[1:]>UB), (ls_autocorr_i[1:]<LB))) ==0: # Criteria for kicking out the genes
            n_rejects = n_rejects+1
            # if count <10:
            #     count = count + 1
            #     f,a = plt.subplots(nrows=2,ncols=1)
            #     a[0].plot(data_i.T)
            #     a[1].stem(ls_autocorr_i)
            #     a[1].plot(ls_autocorr_i*0 + UB,color ='red')
            #     a[1].plot(ls_autocorr_i * 0 + LB,color ='red')
            #     f.show()
        else:
            ls_filtered_genes.append(ls_genes[i])
            # if count <10:
            #     count = count + 1
            #     f,a = plt.subplots(nrows=2,ncols=1)
            #     a[0].plot(data_i.T)
            #     a[1].stem(ls_autocorr_i)
            #     a[1].plot(ls_autocorr_i*0 + UB,color ='red')
            #     a[1].plot(ls_autocorr_i * 0 + LB,color ='red')
            #     f.show()
    print('No. of genes thrown out by flatness test (autocorrelation of residuals): ',n_rejects)
    dict_out = copy.deepcopy(dict_data_in)
    for cond, rep in itertools.product(ls_conditions, ls_replicates):
        dict_out[cond][rep]['df_X_TPM'] = dict_data_in[cond][rep]['df_X_TPM'].loc[ls_filtered_genes,:]
    return dict_out

ls_conditions = list(dict_DATA_ORIGINAL.keys())  # condition identifiers
ls_replicates = list(dict_DATA_ORIGINAL[ls_conditions[0]].keys())  # replicate identifiers
ls_replicates.sort()
ngenes = dict_DATA_ORIGINAL[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].shape[0]  # number of genes
ls_genes = list(dict_DATA_ORIGINAL[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].index)  # gene list
#
## DO THE EQUILIBRIUM REMOVAL - Shara: Shifting the curves by a constant value should not affect the flatness of the data
# TODO - considering all the replicates, in future, we need to change it to the training replicate
if NORMALIZE_BY_NC_EQUILIBRIUM:
    x_eq_BIAS = 1e-15
else:
    x_eq_BIAS = 0
if REMOVE_NC_EQUILIBRIUM:
    x_eq = np.zeros(shape=(ngenes, 0))
    for rep_i in range(16):
        x_eq = np.concatenate([x_eq, np.array(dict_DATA_ORIGINAL['NC'][rep_i]['df_X_TPM'].iloc[:, -1:])], axis=1)
    x_eq = x_eq.mean(axis=1).reshape(-1, 1) + x_eq_BIAS
    if NORMALIZE_BY_NC_EQUILIBRIUM:
        for cond, rep_i in itertools.product(ls_conditions, range(16)):
            dict_DATA_ORIGINAL[cond][rep_i]['df_X_TPM'] = dict_DATA_ORIGINAL[cond][rep_i]['df_X_TPM'].subtract(x_eq, axis=0).divide(x_eq, axis=0)
    else:
        for cond, rep_i in itertools.product(ls_conditions, range(16)):
            dict_DATA_ORIGINAL[cond][rep_i]['df_X_TPM'] = dict_DATA_ORIGINAL[cond][rep_i]['df_X_TPM'].subtract(x_eq, axis=0)


## Flatness test
# dict_DATA_FILTERED = filter_genes_by_flatness_test(dict_DATA_ORIGINAL, significance_level_percent= 1e-10)
dict_DATA_FILTERED = filter_genes_by_flatness_test(dict_DATA_ORIGINAL, significance_level_percent= 5)
ngenes_filtered = dict_DATA_FILTERED['MX'][0]['df_X_TPM'].shape[0]
ls_locus_tags_filtered = list(dict_DATA_FILTERED['MX'][0]['df_X_TPM'].index)
print('Number of filtered genes: ', ngenes_filtered)

# Genes in comparison to DESeq2
# df_DE = pd.read_csv('/Users/shara/Desktop/masigpro_RNAseq/DESEq_ordered_by_adjusted_p_values.csv',index_col=0)
# df_DE_filt = df_DE[df_DE['padj']<1e-15] # Original
# # df_DE_filt = df_DE[df_DE['padj']<1e-1]
# ls_DEfiltered_genes = list(df_DE_filt.index)

# Genes in comparison to DESeq2
# df_DE = pd.read_csv('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/DE_genes_0_585.csv',index_col=0)
df_DE = pd.read_csv('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/DE_genes_1_0.csv',index_col=0)
ls_DEfiltered_genes = list(df_DE.loc[:,'DEgenes'])

ls_autocorrelation_and_DE_intersection = list(set(ls_locus_tags_filtered) & set(ls_DEfiltered_genes))
# ls_autocorrelation_and_DE_intersection = ls_locus_tags_filtered
print('No of genes filtered by autocorrelation: ', ngenes_filtered)
print('No of genes filtered by DESeq2: ', len(ls_DEfiltered_genes))
print('No of intersecting genes: ',len(ls_autocorrelation_and_DE_intersection))
dict_DATA_FILTERED = copy.deepcopy(dict_DATA_ORIGINAL)
for cond, rep in itertools.product(ls_conditions, ls_replicates):
    dict_DATA_FILTERED[cond][rep]['df_X_TPM'] = dict_DATA_FILTERED[cond][rep]['df_X_TPM'].loc[ls_autocorrelation_and_DE_intersection, :]

f,ax = plt.subplots(nrows = 3,ncols=2, sharex= True, sharey=True, figsize=(12,18))
ax = ax.reshape(-1)
ax[0].plot(np.array(dict_DATA_ORIGINAL['MX'][5]['df_X_TPM']).T)
ax[0].set_title('MAX Original')
ax[1].plot(np.array(dict_DATA_FILTERED['MX'][5]['df_X_TPM']).T)
ax[1].set_title('Filtered MAX \n (Flatness test)')
ax[2].plot(np.array(dict_DATA_ORIGINAL['NC'][5]['df_X_TPM']).T)
ax[2].set_title('NC Original')
ax[3].plot(np.array(dict_DATA_FILTERED['NC'][5]['df_X_TPM']).T)
ax[3].set_title('Filtered NC \n (Flatness test)')
ax[4].plot(np.array(dict_DATA_ORIGINAL['MN'][5]['df_X_TPM']).T)
ax[4].set_title('MIN Original')
ax[5].plot(np.array(dict_DATA_FILTERED['MN'][5]['df_X_TPM']).T)
ax[5].set_title('Filtered MIN \n (Flatness test)')
f.show()

ngenes_filtered = dict_DATA_FILTERED['MX'][0]['df_X_TPM'].shape[0]
ls_locus_tags_filtered = list(dict_DATA_FILTERED['MX'][0]['df_X_TPM'].index)
n_outputs = np.shape(dict_DATA_FILTERED['NC'][1]['Y'])[0]


## DO THE EQUILIBRIUM REMOVAL
# TODO - considering all the replicates, in future, we need to change it to the training replicate
if REMOVE_NC_EQUILIBRIUM:
    x_eq = np.zeros(shape=(ngenes_filtered,0))
    if SUBTRACT_FINAL_TIME_POINT:
        y_eq = np.zeros(shape=(1, 0))
    else:
        y_eq = np.zeros(shape=(n_outputs,0))
    for rep_i  in range(16):
        x_eq = np.concatenate([x_eq,np.array(dict_DATA_FILTERED['NC'][rep_i]['df_X_TPM'].iloc[:, -1:])],axis=1)
        if SUBTRACT_FINAL_TIME_POINT:
            y_eq = np.concatenate([y_eq, np.array(dict_DATA_FILTERED['NC'][rep_i]['Y'].iloc[-1:, -1:])], axis=1)
        else:
            y_eq = np.concatenate([y_eq, np.array(dict_DATA_FILTERED['NC'][rep_i]['Y'].iloc[:, -1:])], axis=1)
    x_eq = x_eq.mean(axis=1).reshape(-1,1) + x_eq_BIAS
    y_eq = y_eq.mean(axis=1).reshape(-1,1)
    for cond, rep_i in itertools.product(ls_conditions,range(16)):
        if NORMALIZE_BY_NC_EQUILIBRIUM:
            dict_DATA_FILTERED[cond][rep_i]['df_X_TPM'] = dict_DATA_FILTERED[cond][rep_i]['df_X_TPM'].subtract(x_eq, axis=0).divide(x_eq, axis=0)
        else:
            dict_DATA_FILTERED[cond][rep_i]['df_X_TPM'] = dict_DATA_FILTERED[cond][rep_i]['df_X_TPM'].subtract(x_eq,axis=0)
        if SUBTRACT_FINAL_TIME_POINT:
            dict_DATA_FILTERED[cond][rep_i]['Y'] = dict_DATA_FILTERED[cond][rep_i]['Y'] - y_eq[0,0]
        else:
            dict_DATA_FILTERED[cond][rep_i]['Y'] = dict_DATA_FILTERED[cond][rep_i]['Y'].subtract(y_eq, axis=0)
# try:
# if N_CONCATENATED_OUTPUTS ==1:
#
# ls_time = np.arange(0,len(np.array(dict_DATA_FILTERED['MX'][1]['Y']).reshape(-1)))* 3/60
# plt.plot(ls_time,np.array(dict_DATA_FILTERED['MX'][1]['Y']).T.reshape(-1))
# plt.plot(ls_time[40:],np.array(dict_DATA_FILTERED['MN'][1]['Y']).T.reshape(-1))
# plt.plot(ls_time[40:],np.array(dict_DATA_FILTERED['NC'][1]['Y']).T.reshape(-1))
# plt.title('Output curves \n (NC Equilibrium subtracted)')
# plt.xlabel('Time (hrs)')
# plt.show()
# except:
#  print('No plotting the outputs')

f,ax = plt.subplots(nrows = 3,ncols=1, sharex= True, sharey=True, figsize=(12,18))
ax[0].plot(np.array(dict_DATA_FILTERED['MX'][5]['df_X_TPM']).T)
ax[0].set_title('MAX')
ax[1].plot(np.array(dict_DATA_FILTERED['NC'][5]['df_X_TPM']).T)
ax[1].set_title('NC')
ax[2].plot(np.array(dict_DATA_FILTERED['MN'][5]['df_X_TPM']).T)
ax[2].set_title('MIN')
f.show()

##
SYSTEM_NO = 1000
# 'standard' / 'min max' / 'normalizer' / 'none'
X_scale_method = 'standard'#'standard' #'min max'
U_scale_method = 'standard'
Y_scale_method = 'none'
MEAN_SCALER_X = False
MEAN_SCALER_U = False
MEAN_SCALER_Y = False
STD_SCALER_X = True
STD_SCALER_U = True
STD_SCALER_Y = False
rnaf.formulate_and_save_Koopman_Data(dict_DATA_FILTERED, ALL_CONDITIONS=ls_conditions, SYSTEM_NO=SYSTEM_NO, X_method=X_scale_method, U_method=U_scale_method, Y_method=Y_scale_method, WITH_MEAN_FOR_STANDARD_SCALER_X=MEAN_SCALER_X, WITH_MEAN_FOR_STANDARD_SCALER_U=MEAN_SCALER_U, WITH_MEAN_FOR_STANDARD_SCALER_Y=MEAN_SCALER_Y, WITH_STD_FOR_STANDARD_SCALER_X=STD_SCALER_X, WITH_STD_FOR_STANDARD_SCALER_U=STD_SCALER_U, WITH_STD_FOR_STANDARD_SCALER_Y = STD_SCALER_Y)
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'

##
SYSTEM_NO = 1000
ls_conditions = ['MX','MN','NC']
dict_temp = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ls_conditions)

# ls_replicates = dict_temp
XpTs = dict_temp['train']['XpTs']
UpTs = dict_temp['train']['UpTs']
XfTs = dict_temp['train']['XfTs']
YpTs = dict_temp['train']['YpTs']
YfTs = dict_temp['train']['YfTs']

XpTs_v = dict_temp['valid']['XpTs']
UpTs_v = dict_temp['valid']['UpTs']
XfTs_v = dict_temp['valid']['XfTs']
YpTs_v = dict_temp['valid']['YpTs']
YfTs_v = dict_temp['valid']['YfTs']

f,ax = plt.subplots(nrows = 3,ncols=1, sharex= True, sharey=True, figsize=(12,18))
ax[0].plot(np.array(dict_temp['scaled']['MX'][0]['XfT']))
ax[0].set_title('MAX')
ax[1].plot(np.array(dict_temp['scaled']['NC'][0]['XfT']))
ax[1].set_title('NC')
ax[2].plot(np.array(dict_temp['scaled']['MN'][0]['XfT']))
ax[2].set_title('MIN')
f.show()

##
LASSO_FIT = True
# Lasso fit parameters
# multioutput = 'variance_weighted' #'uniform'
multioutput = 'uniform'
REINITIALIZE_PREVIOUS_SOLUTION = False
ONLY_POSITIVE_COEFFICIENTS = False
dict_x_lasso = {}
# ls_regularization_lambda_x = [0,10000,5000,2000,1000,500,200,100,50]
# ls_regularization_lambda_x = [0, 0.005, 0.001, 0.01, 0.05]#,0.1,0.01]
# ls_regularization_lambda_x = [0,1000,100,10,1]
# ls_regularization_lambda_x = [0,0.2,0.5,0.8]
# ls_regularization_lambda_x = [0,0.01,0.02,0.05,0.1]
# ls_regularization_lambda_x = [0.002,0.005,0]
# ls_regularization_lambda_x = [0,0.0001,0.0005,0.001,0.0005]
ls_regularization_lambda_x = [0,0.001]

if LASSO_FIT:
    for alpha_i in ls_regularization_lambda_x:
        print('alpha =', alpha_i)
        dict_x_lasso[alpha_i] = {}
        # 1 -step Fit
        if alpha_i == 0:
            model_i = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        else:
            # model_i = Lasso(alpha= 1, fit_intercept= False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6, warm_start=REINITIALIZE_PREVIOUS_SOLUTION, positive=ONLY_POSITIVE_COEFFICIENTS)
            model_i = Lasso(alpha=alpha_i, fit_intercept=False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6,
                            warm_start=REINITIALIZE_PREVIOUS_SOLUTION, positive=ONLY_POSITIVE_COEFFICIENTS)
            # model_i = Lasso(alpha=alpha_i, fit_intercept=False, normalize=False, max_iter=50000)
        model_i.fit(np.concatenate([XpTs, UpTs], axis=1), XfTs)
        dict_x_lasso[alpha_i]['model'] = model_i
        # 1-step prediction error
        dict_x_lasso[alpha_i]['train_1step_error'] = r2_score(XfTs,model_i.predict(np.concatenate([XpTs,UpTs],axis=1)),multioutput='variance_weighted')
        dict_x_lasso[alpha_i]['valid_1step_error'] = r2_score(XfTs_v, model_i.predict(np.concatenate([XpTs_v, UpTs_v], axis=1)),multioutput='variance_weighted')
        dict_x_lasso[alpha_i]['train_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='train', multioutput=multioutput)
        dict_x_lasso[alpha_i]['valid_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='valid', multioutput=multioutput)

    df_lasso_stats_x = pd.DataFrame(dict_x_lasso).iloc[1:,:]
    lasso_opt_stats_x = df_lasso_stats_x .loc[:,df_lasso_stats_x.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0) == np.max(df_lasso_stats_x.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0))]
    print('Optimal alpha and its LASSO regression statistics for X')
    print(lasso_opt_stats_x)
    lasso_X_model = dict_x_lasso[np.array([lasso_opt_stats_x.columns])[0, 0]]['model']

## Lasso Fit Y
dict_y_lasso = {}
# ls_regularization_lambda_y = [0,1,1e-1,1e-2]
ls_regularization_lambda_y = [0,0.001]
# ls_regularization_lambda_y = [0,1,0.1,0.01,1e-3]#,5e-3,1e-4,5e-4]
# ls_regularization_lambda_y = [2e-4]
multioutput = 'uniform'

if LASSO_FIT:
    print(' Lasso Regression Fit of the output Y equation ')
    for alpha_i in ls_regularization_lambda_y:
        print('alpha =', alpha_i)
        dict_y_lasso[alpha_i] = {}
        # 1 -step Fit
        if alpha_i == 0:
            model_i = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        else:
            # model_i = Lasso(alpha= 1, fit_intercept= False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6, warm_start=REINITIALIZE_PREVIOUS_SOLUTION, positive=ONLY_POSITIVE_COEFFICIENTS)
            model_i = Lasso(alpha=alpha_i, fit_intercept=False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6,
                            warm_start=REINITIALIZE_PREVIOUS_SOLUTION, positive=ONLY_POSITIVE_COEFFICIENTS)
            # model_i = Lasso(alpha=alpha_i, fit_intercept=False, normalize=False, max_iter=50000)
        model_i.fit(np.concatenate([XpTs, XfTs], axis=0), np.concatenate([YpTs, YfTs], axis=0))
        dict_y_lasso[alpha_i]['model'] = model_i
        # 1-step prediction error
        dict_y_lasso[alpha_i]['train_1step_error'] = r2_score(np.concatenate([YpTs, YfTs], axis=0).reshape(-1),model_i.predict(np.concatenate([XpTs, XfTs], axis=0)).reshape(-1))
        dict_y_lasso[alpha_i]['valid_1step_error'] = r2_score(np.concatenate([YpTs, YfTs], axis=0).reshape(-1),model_i.predict(np.concatenate([XpTs, XfTs], axis=0)).reshape(-1))
        dict_y_lasso[alpha_i]['train_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='train', multioutput=multioutput)
        dict_y_lasso[alpha_i]['valid_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='valid', multioutput=multioutput)

    df_lasso_stats_y = pd.DataFrame(dict_y_lasso).iloc[1:,:]
    lasso_opt_stats_y = df_lasso_stats_y .loc[:,df_lasso_stats_y.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0) == np.max(df_lasso_stats_y.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0))]
    print('Optimal alpha and its LASSO Regression Statistics for Y')
    print(lasso_opt_stats_y)
    lasso_Y_model = dict_y_lasso[np.array([lasso_opt_stats_y.columns])[0,0]]['model']

##
RIDGE_FIT = True
multioutput = 'variance_weighted'
# ls_regularization_lambda_x = [0,1,1e-1,1e-2,1e-3]
# ls_regularization_lambda_x = list(np.arange(0,100,2))
ls_regularization_lambda_x = list(np.arange(0,2,0.1))
# ls_regularization_lambda_x = list(np.arange(0,3,0.1))
# ls_regularization_lambda_x.extend([2,3,4,5,6,7,8,9,10])
# ls_regularization_lambda_y = list(np.arange(0,100,2))
ls_regularization_lambda_y = list(np.arange(0,2,0.1))
# ls_regularization_lambda_y = list(np.arange(0,4,0.1))
dict_x_ridge = {}
dict_y_ridge = {}
if RIDGE_FIT:
    for alpha_i in ls_regularization_lambda_x:
        print('alpha =', alpha_i)
        dict_x_ridge[alpha_i] = {}
        # 1 -step Fit
        if alpha_i == 0:
            model_i = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        else:
            # model_i = Lasso(alpha= 1, fit_intercept= False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6, warm_start=REINITIALIZE_PREVIOUS_SOLUTION, positive=ONLY_POSITIVE_COEFFICIENTS)
            model_i = Ridge(alpha=alpha_i, fit_intercept=False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6)
            # model_i = Lasso(alpha=alpha_i, fit_intercept=False, normalize=False, max_iter=50000)
        model_i.fit(np.concatenate([XpTs, UpTs], axis=1), XfTs)
        dict_x_ridge[alpha_i]['model'] = model_i
        # 1-step prediction error
        dict_x_ridge[alpha_i]['train_1step_error'] = r2_score(XfTs,model_i.predict(np.concatenate([XpTs,UpTs],axis=1)),multioutput='variance_weighted')
        dict_x_ridge[alpha_i]['valid_1step_error'] = r2_score(XfTs_v, model_i.predict(np.concatenate([XpTs_v, UpTs_v], axis=1)),multioutput='variance_weighted')
        dict_x_ridge[alpha_i]['train_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='train', multioutput=multioutput)
        dict_x_ridge[alpha_i]['valid_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='valid', multioutput=multioutput)

    df_ridge_stats_x = pd.DataFrame(dict_x_ridge).iloc[1:,:]
    ridge_opt_stats_x = df_ridge_stats_x.loc[:,df_ridge_stats_x.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0) == np.max(df_ridge_stats_x.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0))]
    print('Optimal alpha and its Ridge regression statistics for state model (X)')
    print(ridge_opt_stats_x)
    ridge_X_model = dict_x_ridge[np.array([ridge_opt_stats_x.columns])[0,0]]['model']

    for alpha_i in ls_regularization_lambda_y:
        print('alpha =', alpha_i)
        dict_y_ridge[alpha_i] = {}
        # 1 -step Fit
        if alpha_i == 0:
            model_i = LinearRegression(fit_intercept=False, normalize=False, copy_X=True)
        else:
            model_i = Ridge(alpha=alpha_i, fit_intercept=False, normalize=False, copy_X=True, max_iter=50000, tol=1e-6)
        model_i.fit(np.concatenate([XpTs, XfTs], axis=0), np.concatenate([YpTs, YfTs], axis=0))
        dict_y_ridge[alpha_i]['model'] = model_i
        # 1-step prediction error
        dict_y_ridge[alpha_i]['train_1step_error'] = r2_score(np.concatenate([YpTs, YfTs], axis=0).reshape(-1), model_i.predict(np.concatenate([XpTs,XfTs],axis=0)).reshape(-1))
        dict_y_ridge[alpha_i]['valid_1step_error'] = r2_score(np.concatenate([YpTs_v, YfTs_v], axis=0).reshape(-1), model_i.predict(np.concatenate([XpTs_v,XfTs_v],axis=0)).reshape(-1))
        dict_y_ridge[alpha_i]['train_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='train', multioutput=multioutput)
        dict_y_ridge[alpha_i]['valid_nstep_error'] = n_step_prediction_error(model_i, dict_temp, train_test_valid='valid', multioutput=multioutput)

    df_ridge_stats_y = pd.DataFrame(dict_y_ridge).iloc[1:,:]
    ridge_opt_stats_y = df_ridge_stats_y.loc[:,df_ridge_stats_y.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0) == np.max(df_ridge_stats_y.loc[['train_nstep_error','valid_nstep_error'],:].sum(axis=0))]
    print('Optimal alpha and its Ridge regression statistics for output model (Y)')
    print(ridge_opt_stats_y)
    ridge_Y_model = dict_y_ridge[np.array([ridge_opt_stats_y.columns])[0,0]]['model']



## Fitting X
X_scaler = dict_temp['X_scaler']
U,s,VT = np.linalg.svd(np.concatenate([XpTs,UpTs],axis=1).T)
# U,s,VT = np.linalg.svd(XpTs.T)

# np.min(XpTs.shape)
# for r in range(10,200,5):
for r in range(13, 14, 1): #33
# for r in range(15, 16, 1): #33
# for r in range(41, 42, 1): # Experimenting
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Ahat = XfTs.T @ Vr @ np.linalg.inv(Sr) @ UrT
    # sb.heatmap(Ahat,cmap='RdBu')
    # plt.show()
    XfT_true = np.empty(shape=(0,ngenes_filtered))
    XfT_est = np.empty(shape=(0,ngenes_filtered))
    for cond,rep in itertools.product(ls_conditions,dict_temp['valid']['indices']):
        # Predict the Xf - n step
        XfTsn_hat = dict_temp['scaled'][cond][rep]['XpT'][0:1, :]
        for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
            XfTsn_hat = np.concatenate([XfTsn_hat, np.concatenate([XfTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T], axis=0)
            # XfTsn_hat = np.concatenate([XfTsn_hat, XfTsn_hat[-1:] @ Ahat.T],axis=0)
        XfTsn_hat = XfTsn_hat[1:]
        XfT_true = np.concatenate([XfT_true, X_scaler.inverse_transform(dict_temp['scaled'][cond][rep]['XfT'])], axis=0)
        XfT_est = np.concatenate([XfT_est, X_scaler.inverse_transform(XfTsn_hat)], axis=0)
    print('r = ',r,' | Validation Data r^2 val = ',r2_score(XfT_true.reshape(-1), XfT_est.reshape(-1)))

## n-step predictions
# Predict on all the datasets
dict_DATA_PREDICTED = {}
for cond in ls_conditions:
    dict_DATA_PREDICTED[cond] = {}
XfTs_final_true = np.empty(shape=(0,ngenes_filtered))
XfTs_final_est = np.empty(shape=(0,ngenes_filtered))
# for cond,rep in itertools.product(ls_conditions,dict_temp['valid']['indices']):
for cond, rep in itertools.product(ls_conditions, ls_replicates):
    # Predict the Xf - n step
    XfTsn_hat = dict_temp['scaled'][cond][rep]['XpT'][0:1,:]
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XfTsn_hat = np.concatenate([XfTsn_hat,np.concatenate([XfTsn_hat[-1:],dict_temp['scaled'][cond][rep]['UpT'][-1:]],axis=1) @ Ahat.T],axis=0)
        # XfTsn_hat = np.concatenate([XfTsn_hat, XfTsn_hat[-1:] @ Ahat.T],axis=0)
    if cond == 'MX':
        ls_time =[1,2,3,4,5,6,7]
    else:
        ls_time =[3,4,5,6,7]
    dict_DATA_PREDICTED[cond][rep]={'df_X_TPM': pd.DataFrame(X_scaler.inverse_transform(XfTsn_hat).T,index=ls_locus_tags_filtered,columns=ls_time)}
    XfTsn_hat = XfTsn_hat[1:]
    XfTs_final_true = np.concatenate([XfTs_final_true, dict_temp['scaled'][cond][rep]['XfT'][-1:]],axis=0)
    XfTs_final_est = np.concatenate([XfTs_final_est, XfTsn_hat[-1:]], axis=0)
# plt.plot(XfTs_final_true[0:16].reshape(-1),XfTs_final_est[0:16].reshape(-1),'.')
# plt.plot(XfTs_final_true[16:32].reshape(-1),XfTs_final_est[16:32].reshape(-1),'.')
# plt.plot(XfTs_final_true[32:].reshape(-1),XfTs_final_est[32:].reshape(-1),'.')
# plt.show()
print('Error in n-step prediction :', r2_score(XfTs_final_true.reshape(-1),XfTs_final_est.reshape(-1)))

## Fitting the output Y

U,s,VT = np.linalg.svd(np.concatenate([XpTs,XfTs],axis=0).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
U_scaler = dict_temp['U_scaler']
Y_scaler = dict_temp['Y_scaler']
 #np.min(XpTs.shape)
# for r in range(1,60,1): #opt r = 33
for r in range(13, 14, 1):  # opt r = 27
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Chat = np.concatenate([YpTs,YfTs],axis=0).T @ Vr @ np.linalg.inv(Sr) @ UrT
    # sb.heatmap(Chat,cmap='RdBu')
    # plt.show()

    YT_true = np.empty(shape=(0,N_CONCATENATED_OUTPUTS))
    YT_est = np.empty(shape=(0,N_CONCATENATED_OUTPUTS))
    for cond,rep in itertools.product(ls_conditions,dict_temp['train']['indices']):
        XTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['XpT'][0:1],dict_temp['scaled'][cond][rep]['XfT']],axis=0)
        YTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
        YTs_hat = XTs_all @ Chat.T
        YT_true = np.concatenate([YT_true, Y_scaler.inverse_transform(YTs_all)], axis=0)
        YT_est = np.concatenate([YT_est, Y_scaler.inverse_transform(YTs_hat)], axis=0)
    print('r = ',r,' | Validation Data r^2 val = ',r2_score(YT_true.reshape(-1), YT_est.reshape(-1)))

## TODO Temporary fix for transition --- fix this later
dict_models ={}
dict_models['DMD'] = {'A': copy.deepcopy(Ahat[:,:-2]),'B': copy.deepcopy(Ahat[:,-2:]), 'C': copy.deepcopy(Chat)}
try:
    dict_models['lasso'] = {'A': lasso_X_model.coef_[:,:-2],'B': lasso_X_model.coef_[:,-2:], 'C': lasso_Y_model.coef_}
except:
    print('[WARNING]: Lasso Regression not done')
try:
    dict_models['ridge'] = {'A': ridge_X_model.coef_[:,:-2],'B': ridge_X_model.coef_[:,-2:], 'C': ridge_Y_model.coef_}
except:
    print('[WARNING]: Ridge Regression not done')

##
# for model_choice,FITNESS
# model_choice = 'DMD'
model_choice = 'lasso'
# model_choice = 'ridge'

FITNESS_DEFECT = True
# FITNESS_DEFECT = False

N_GENES_TO_PLOT = 10

# Prediction options
SCALED = False
COMPUTE_OD = True
N_STEP = True
# Plot options
WITH_EMBEDDING = False
PLOT_GRAY = False


Ahat = copy.deepcopy(np.concatenate([dict_models[model_choice]['A'],dict_models[model_choice]['B']],axis=1))
if N_CONCATENATED_OUTPUTS == 1:
    Chat = copy.deepcopy(dict_models[model_choice]['C']).reshape(1,-1)
else:
    Chat = copy.deepcopy(dict_models[model_choice]['C'])


ls_filtered_gene_nos = np.arange(ngenes_filtered)
set_filtered_gene_nos = set(ls_filtered_gene_nos)


rep = dict_temp['test']['indices'][0]
cond = 'MX'
YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
YT_true = Y_scaler.inverse_transform(YTs_true)
XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
    XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
YTsn_hat = XTsn_hat @ Chat.T
YTn_hat0 = Y_scaler.inverse_transform(YTsn_hat)

dict_gene_select_r2 = {}
dict_gene_select_fc = {}
epsilon = 1e-3

# simluate single gene knockouts
ls_gene_select = list(set_filtered_gene_nos)
for i in range(len(ls_gene_select)):
    try:
        XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
        XTsn_hat[0, i] = 0
        for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
            XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T], axis=0)
            XTsn_hat[-1, i] = 0
        YTsn_hat = XTsn_hat @ Chat.T
        YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
        # dict_gene_select[ls_gene_select] = r2_score(YT_true,YTn_hat)
        # dict_gene_select_r2[tuple([ls_gene_select[i]])] = r2_score(YTn_hat0, YTn_hat)
        dict_gene_select_r2[tuple([ls_gene_select[i]])] = np.sum(YTn_hat.reshape(-1) - YTn_hat0.reshape(-1))

        # if FITNESS_DEFECT:
        #     dict_gene_select_r2[tuple([ls_gene_select[i]])] = np.sum(YTn_hat0.reshape(-1) - YTn_hat.reshape(-1))
        #     # dict_gene_select_r2[tuple([ls_gene_select[i]])] = - np.sum(np.maximum(0, YTn_hat0.reshape(-1) - YTn_hat.reshape(-1)))
        # else:
        #     dict_gene_select_r2[tuple([ls_gene_select[i]])] = np.sum(YTn_hat0.reshape(-1) - YTn_hat.reshape(-1))
        #     # dict_gene_select_r2[tuple([ls_gene_select[i]])] = np.sum(np.minimum(0, YTn_hat.reshape(-1) - YTn_hat0.reshape(-1)))
        # # dict_gene_select_fc[tuple([ls_gene_select[i]])] = np.median(
        # #     np.abs(1 - ((YTn_hat[:, 0] + epsilon) / (YTn_hat0[:, 0] + epsilon))))
        #
        # # if np.mod(count,100000) ==0:
        # #     # print (count,'/',total_choices,' complete')
        # #     break
    except:
        break


if FITNESS_DEFECT:
    sort_reverse = False
    cmap = plt.get_cmap('Reds')
else:
    sort_reverse = True
    cmap = plt.get_cmap('Greens')
cmaplist = [cmap(i) for i in range(cmap.N)]
N_start = 10
N_finish = 250
ls_color_index = np.arange(N_start,N_finish,np.int(np.floor((N_finish-N_start)/N_GENES_TO_PLOT)))
ls_color_index = np.flip(ls_color_index)
ls_colors = [cmaplist[i] for i in ls_color_index]
sorted_dict_r2 = dict(sorted(dict_gene_select_r2.items(), reverse=sort_reverse, key=operator.itemgetter(1)))
# sorted_dict_fc = dict(sorted(dict_gene_select_fc.items(), reverse=sort_reverse, key=operator.itemgetter(1))) # Sorted by fold change (FC)


# Enoch - validate the gene list by predicting the output curves by generating
# cond = 'MX'

Y0 = dict_DATA_ORIGINAL[cond][rep]['Y0']
ls_ordered_indices = []
ls_ordered_genes = []
np_dist = np.array([])
for items in list(sorted_dict_r2.keys()):
# for items in list(sorted_dict_r2.keys())[0:N_GENES_TO_PLOT]:
    ls_ordered_indices.extend([items[0]])
    ls_ordered_genes.extend([ls_locus_tags_filtered[i] for i in list(items)])
    np_dist = np.concatenate([np_dist, np.array([sorted_dict_r2[items]])],axis=0)

import time
start = time.time()
# Wild type strain behavior under prediction
XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
    if N_STEP:
        XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T], axis=0)
    else:
        x_Ti = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][j:j+1, :])
        XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([x_Ti, dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T], axis=0)
if not SCALED:
    YTsn_hat = XTsn_hat @ Chat.T
    YTn_hat_wild = Y_scaler.inverse_transform(YTsn_hat)
    YT_true = np.concatenate([dict_temp['unscaled'][cond][rep]['YpT'], dict_temp['unscaled'][cond][rep]['YfT'][-1:,:]],axis=0)
    if COMPUTE_OD:
        if REMOVE_NC_EQUILIBRIUM and SUBTRACT_FINAL_TIME_POINT:
            YT_true = YT_true + y_eq[0,0]
            YTn_hat_wild = YTn_hat_wild + y_eq[0, 0]
        elif REMOVE_NC_EQUILIBRIUM:
            YT_true = YT_true + y_eq.T
            YTn_hat_wild = YTn_hat_wild + y_eq.T
        YT_true = 2 ** YT_true * Y0
        YTn_hat_wild = 2 ** YTn_hat_wild * Y0
    max_y_val = np.max([np.max(YT_true), np.max(YTn_hat_wild)])
else:
    YTsn_hat_wild = XTsn_hat @ Chat.T
    YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'], dict_temp['scaled'][cond][rep]['YfT'][-1:,:]],axis=0)
end = time.time()

print('Total simulation time: ', end - start)

##
if WITH_EMBEDDING:
    MARKER_SIZE = 0
else:
    MARKER_SIZE = 15


plt.figure(figsize = (8.5,6))
ls_time = np.arange(0,len(YT_true.reshape(-1)))* 3/60
# Mutant strains (negative step input)
for i in range(len(ls_ordered_indices[0:N_GENES_TO_PLOT])):
    b = np.zeros(shape=(len(ls_gene_select),1))
    b[ls_ordered_indices[i],0] = 1
    XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
    XTsn_hat[0, ls_ordered_indices[i]] = 0
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        if N_STEP:
            XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T], axis=0)
        else:
            x_Ti = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][j:j + 1, :])
            XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([x_Ti, dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T], axis=0)
        XTsn_hat[-1, ls_ordered_indices[i]] = 0
    YTsn_hat = XTsn_hat @ Chat.T
    if not SCALED:
        YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
        # dist = np.round(np.sum(YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1)), 2)
        dist = np.round(np_dist[i],2)
        if COMPUTE_OD:
            if REMOVE_NC_EQUILIBRIUM and SUBTRACT_FINAL_TIME_POINT:
                YTn_hat = YTn_hat + y_eq[0, 0]
            elif REMOVE_NC_EQUILIBRIUM:
                YTn_hat = YTn_hat + y_eq.T
            YTn_hat = 2 ** YTn_hat * Y0
        if FITNESS_DEFECT:
            # dist = np.round(np.sqrt(np.sum(np.maximum(0, YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1)) ** 2)), 2)
            # dist = np.round(np.sum(np.maximum(0, YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1))), 2)
            # dist = np.round(np.sum(YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1)),2)
            symbol = '-'
            dist = np.abs(dist)
        else:
            # dist = np.round(np.sqrt(np.sum(np.minimum(0, YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1)) ** 2)), 2)
            # dist = np.round(np.sum(np.minimum(0, YTn_hat.reshape(-1) - YTn_hat_wild.reshape(-1))), 2)
            # dist = np.round(np.sum(YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1)), 2)
            symbol = '+'
        # Plotting based on options
        if WITH_EMBEDDING and PLOT_GRAY:
            plt.plot(ls_time, YTn_hat.reshape(-1), color = 'gray')
        elif WITH_EMBEDDING:
            plt.plot(ls_time, YTn_hat.reshape(-1), label=ls_ordered_genes[i] + '( ' + symbol + ' ' + str(dist) + ')', linewidth=2, color=ls_colors[i], marker='.', markersize=MARKER_SIZE)
        elif PLOT_GRAY:
            plt.plot(np.arange(1, 8), YTn_hat[:, 0].reshape(-1), color='gray')
        else:
            plt.plot(np.arange(1, 8), YTn_hat[:, 0], label=ls_ordered_genes[i] + '( ' + symbol + ' ' + str(dist) + ')', linewidth=2, color=ls_colors[i], marker='.', markersize=MARKER_SIZE)
        max_y_val = np.max([max_y_val, np.max(YTn_hat)])
    else:
        if COMPUTE_OD:
            print('Wrong stat for scaling. Scaling = False for OD computation')
        dist = np.round(np.sqrt(np.sum(np.maximum(0, YTsn_hat_wild.reshape(-1) - YTsn_hat.reshape(-1)) ** 2)), 2)
        # Plotting based on options
        if WITH_EMBEDDING and PLOT_GRAY:
            plt.plot(ls_time, YTsn_hat.reshape(-1), color = 'gray')
        elif WITH_EMBEDDING:
            plt.plot(ls_time, YTsn_hat.reshape(-1), label=ls_ordered_genes[i] + '( - ' + str(dist) + ')', linewidth=2, color=ls_colors[i])
        elif PLOT_GRAY:
            plt.plot(np.arange(1, 8), YTsn_hat[:, 0], color='gray')
        else:
            plt.plot(np.arange(1, 8), YTsn_hat[:, 0], label=ls_ordered_genes[i] + '( - ' + str(dist) + ')', linewidth=2, color=ls_colors[i])


if WITH_EMBEDDING and SCALED:
    plt.plot(ls_time, YTs_true.reshape(-1), ':', label='Wild (data)', marker='.', markersize=MARKER_SIZE, color='tab:gray', lw=3)
    plt.plot(ls_time, YTsn_hat_wild.reshape(-1), label='Wild (model)', marker='.', markersize=MARKER_SIZE, color='tab:gray')
elif WITH_EMBEDDING:
    plt.plot(ls_time, YT_true.reshape(-1), ':', label='Wild (data)', marker='.', markersize=MARKER_SIZE, color='tab:gray', lw=3)
    plt.plot(ls_time, YTn_hat_wild.reshape(-1), label='Wild (model)', marker='.', markersize=MARKER_SIZE, color='tab:gray')
elif SCALED:
    plt.plot(np.arange(1, 8), YTs_true[:, 0], ':', label='Wild (data)', marker='.', markersize=MARKER_SIZE, color='tab:gray', lw=3)
    plt.plot(np.arange(1, 8), YTsn_hat_wild[:, 0], label='Wild (model)', marker='.', markersize=MARKER_SIZE, color='tab:gray')
else:
    plt.plot(np.arange(1, 8), YT_true[:, 0], ':', label='Wild (data)', marker='.', markersize=MARKER_SIZE, color='tab:gray', lw=3)
    plt.plot(np.arange(1, 8), YTn_hat_wild[:, 0], label='Wild (model)', marker='.', markersize=MARKER_SIZE, color='tab:gray')



# plt.legend(loc = 'lower right',ncol = 2,fontsize = 16)
plt.legend(loc = 'upper left',ncol = 1,fontsize = 14)
plt.xlabel('Time (hrs)')
if COMPUTE_OD:
    plt.ylabel('$OD_{600}$')
    # if FITNESS_DEFECT:
    #     plt.ylim(0, 1.5)
    # else:
    #     plt.ylim(0, 1.6)
    plt.ylim(0, 1.1*max_y_val)
else:
    plt.ylabel('$log_2$ fitness')
    plt.ylim(-4, 1)
# plt.title('Scaled output')
# plt.title('')
plt.xlim([0.5,7.1])
plt.savefig('/Users/shara/Desktop/result_temp/RNASeq.svg', transparent= True, bbox_inches='tight')
plt.show()

# Validation by RbTnSeq data
norm = 2
with_respect_to_time0=True
# with_respect_to_time0=False
if cond == 'MX':
    df_RbTnSeq = get_RbTnSeq_curves(ls_ordered_genes[0:N_GENES_TO_PLOT], condition='MAX', with_respect_to_time0=with_respect_to_time0)
elif cond == 'MN':
    df_RbTnSeq = get_RbTnSeq_curves(ls_ordered_genes[0:N_GENES_TO_PLOT], condition='MIN', with_respect_to_time0=with_respect_to_time0)
else:
    df_RbTnSeq = get_RbTnSeq_curves(ls_ordered_genes[0:N_GENES_TO_PLOT], condition='NC', with_respect_to_time0=with_respect_to_time0)
X = np.array(copy.deepcopy(df_RbTnSeq))
X = sg_filter(X,window_length=7,polyorder=3,axis=1)
df_RbTnSeq = pd.DataFrame(X, index=df_RbTnSeq.index, columns=df_RbTnSeq.columns)
# max_val_RbTnSeq = np.max(df_RbTnSeq)
# min_val_RbTnSeq = np.min(df_RbTnSeq)
max_val_RbTnSeq = 0.585
min_val_RbTnSeq = -0.585
plt.figure(figsize=(10,6))
for i in range(N_GENES_TO_PLOT):
    plt.plot(df_RbTnSeq.iloc[i,0:8], marker='.', linewidth=2, markersize=15, color=ls_colors[i])
# plt.legend(df_RbTnSeq.index, loc='lower center', bbox_to_anchor=(0.5, -0.8), ncol=3)
plt.legend(df_RbTnSeq.index, loc='center', bbox_to_anchor=(1.25, 0.5), ncol=1)
plt.plot([0,7],[0,0], color='tab:gray', linewidth=5)
plt.xlabel('Time (hrs)')
plt.ylabel('Relative fitness score')
plt.savefig('/Users/shara/Desktop/result_temp/RbTnSeq.svg', transparent= True, bbox_inches='tight')
plt.show()
# Statistics for the RbTnSeq
# df_RbTnSeq['Defect_dist'] = ((df_RbTnSeq.iloc[:,0:8] - min_val_RbTnSeq)**2).sum(axis=1)**0.5
# df_RbTnSeq['Boost_dist'] = ((df_RbTnSeq.iloc[:,0:8] - max_val_RbTnSeq) ** 2).sum(axis=1) ** 0.5
df_RbTnSeq['Defect_dist'] = np.linalg.norm(np.array(df_RbTnSeq.iloc[:,0:8] - min_val_RbTnSeq),axis=1,ord=norm)
df_RbTnSeq['Boost_dist'] = np.linalg.norm(np.array(df_RbTnSeq.iloc[:,0:8] - max_val_RbTnSeq),axis=1,ord=norm)
df_RbTnSeq['Score'] =  df_RbTnSeq['Defect_dist'] - df_RbTnSeq['Boost_dist'] # Closest genes will have a lower distance
# df_RbTnSeq['Score'] = df_RbTnSeq.iloc[:,-1]
# df_RbTnSeq['Score'] = df_RbTnSeq.sum(axis=1)
print(df_RbTnSeq)
if FITNESS_DEFECT:
    n_RbTnSeq_genes = np.sum(df_RbTnSeq.loc[:,'Score']<0)
    print('% of correctly predicted FITNESS DEFECT genes =')
else:
    n_RbTnSeq_genes = np.sum(df_RbTnSeq.loc[:, 'Score'] > 0)
    print('% of correctly predicted FITNESS BOOST genes =')
n_percent_correct = n_RbTnSeq_genes/N_GENES_TO_PLOT *100
print(n_percent_correct)

## RbTnSeq statistics

condition = 'NC'
df_RbTnSeq = pd.read_csv('DATA/RNA_1_Pput_R2A_Cas_Glu/RbTnSeq/RbTnSeq_' + condition + '.csv', index_col=0).iloc[2:,0:8]
X = np.array(copy.deepcopy(df_RbTnSeq))
X = sg_filter(X,window_length=7,polyorder=3,axis=1)
df_RbTnSeq = pd.DataFrame(X, index=df_RbTnSeq.index, columns=df_RbTnSeq.columns)
# max_val_RbTnSeq = np.max(df_RbTnSeq)
# min_val_RbTnSeq = np.min(df_RbTnSeq)
# max_val_RbTnSeq = np.max(np.max(df_RbTnSeq))/5
# min_val_RbTnSeq = np.min(np.min(df_RbTnSeq))/5
# max_val_RbTnSeq = np.sum(np.maximum(np.array(df_RbTnSeq),0))/np.sum(np.maximum(np.array(df_RbTnSeq),0)>0)
# min_val_RbTnSeq = np.sum(np.minimum(np.array(df_RbTnSeq),0))/np.sum(np.minimum(np.array(df_RbTnSeq),0)<0)

## - - - Hyperparameters for setting thresholds criteria for booster vs defect genes - - - -
##------------------------------------------------------------------------------------------------------------

max_val_RbTnSeq = 0.585  # Anything with a 50% increase in raw relative fitness score is considered a booster gene
min_val_RbTnSeq = -0.585  # Anything with a 50% decrease in raw relative fitness score is considered a fitness defect gene

##------------------------------------------------------------------------------------------------------------
## - - - Hyperparameters for setting thresholds criteria for booster vs defect genes - - - -
# norm = np.inf # 1/2/np.inf
norm = 1
# norm = 2
df_RbTnSeq['Defect_dist'] = np.linalg.norm(np.array(df_RbTnSeq.iloc[:,0:8] - min_val_RbTnSeq),axis=1,ord=norm)
df_RbTnSeq['Boost_dist'] = np.linalg.norm(np.array(df_RbTnSeq.iloc[:,0:8] - max_val_RbTnSeq),axis=1,ord=norm)
df_RbTnSeq['NC_dist'] = np.linalg.norm(np.array(df_RbTnSeq.iloc[:,0:8]),axis=1,ord=norm)
# Idea 1
df_RbTnSeq['min_dist'] = df_RbTnSeq.loc[:,['Defect_dist','Boost_dist','NC_dist']].min(axis=1)
df_RbTnSeq['Classifier'] = -1*(df_RbTnSeq['Defect_dist'] == df_RbTnSeq['min_dist']) + 1*(df_RbTnSeq['Boost_dist'] == df_RbTnSeq['min_dist'])
# plt.plot(np.array(df_RbTnSeq['Classifier']),'.')
# plt.show()
print('% of non NC classified genes = ',100 - np.sum(df_RbTnSeq['Classifier']==0)/len(df_RbTnSeq['Classifier'])*100)
print('# of non NC genes = ', np.sum(df_RbTnSeq['Classifier']!=0))
print('# of BOOSTER genes = ', np.sum(df_RbTnSeq['Classifier']==1))
print('# of DEFECT genes = ', np.sum(df_RbTnSeq['Classifier']==-1))
print('# of NC genes = ', np.sum(df_RbTnSeq['Classifier']==0))
print('P(NC) = ', np.sum(df_RbTnSeq['Classifier']==0)/len(df_RbTnSeq['Classifier']))
print('P(BOOSTER) = ', np.sum(df_RbTnSeq['Classifier']==1)/len(df_RbTnSeq['Classifier']))
print('P(DEFECT) = ', np.sum(df_RbTnSeq['Classifier']==-1)/len(df_RbTnSeq['Classifier']))

#
plt.figure(figsize=(6,6))
plt.plot(np.array(df_RbTnSeq.loc[df_RbTnSeq['Classifier'] ==0,:].iloc[:,0:8]).T,color='tab:gray',alpha = 0.1,markersize=15)
plt.plot(np.array(df_RbTnSeq.loc[df_RbTnSeq['Classifier'] ==1,:].iloc[:,0:8]).T, marker='.',color='tab:green',alpha = 0.45,markersize=15)
plt.plot(np.array(df_RbTnSeq.loc[df_RbTnSeq['Classifier'] ==-1,:].iloc[:,0:8]).T, marker='.',color='tab:red',alpha = 0.1,markersize=15)
plt.plot([],color='tab:gray',label='NC (' + str(np.sum(df_RbTnSeq['Classifier']==0)) +' genes)')
plt.plot([],color='tab:green',label='Boosted (' + str(np.sum(df_RbTnSeq['Classifier']==1)) +' genes)')
plt.plot([],color='tab:red',label='Defect (' + str(np.sum(df_RbTnSeq['Classifier']==-1)) +' genes)')
# plt.legend(fontsize=12, loc='lower center', ncol=3, bbox_to_anchor=(0.0, 1.0))
# plt.legend(fontsize=22, loc='upper center', ncol=1)
plt.legend(loc='upper left', ncol=1, frameon=False)
plt.ylim([-3,5])
plt.xlim([-0.1,7.2])
plt.xticks([0,3.5,7])
# plt.yticks([-3,-1.5,0,1.5,3])
plt.yticks([-2,0,2])
plt.xlabel('Time (hrs)')
plt.ylabel('$\log_2$ Fitness Scores')
ax = plt.gca() # TODO  new command to keep
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.show()

df_RbTnSeq_ref = copy.deepcopy(df_RbTnSeq)


## RNASeq in RbTnSeq
print('No. of NC genes in the RNAseq dataset = ',np.sum(df_RbTnSeq_ref.loc[ls_ordered_genes,'Classifier'] ==0))
print('No. of BOOSTER genes in the RNAseq dataset = ',np.sum(df_RbTnSeq_ref.loc[ls_ordered_genes,'Classifier'] ==1))
print('No. of DEFECT genes in the RNAseq dataset = ',np.sum(df_RbTnSeq_ref.loc[ls_ordered_genes,'Classifier'] ==-1))

##
plt.plot(np.array(df_RbTnSeq['Defect_dist']),'.',label='Defect distance')
plt.plot(np.array(df_RbTnSeq['Boost_dist']),'.',label='Boost distance')
plt.legend()
plt.show()

##
# As = Ahat[:,0:-2]
# Bs = Ahat[:,-2:]
# print('2-norm of A: ',np.linalg.norm(As,2))
# print('2-norm of B: ',np.linalg.norm(Bs,2))

## Export the gene list
dict_out = {}
np_data = np.empty(shape=(0,4))
for items in list(sorted_dict_r2.keys())[0:25]:
    ls_genes_i = [ls_locus_tags_filtered[i]  for i in list(items)]
    entry_i = rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags=ls_genes_i,search_columns='genes(OLN),genes(PREFERRED),protein names,go(biological process)')
    np_data = np.concatenate([np_data,np.array(entry_i).reshape(1,-1)],axis=0)

pd.DataFrame(np_data,columns=['LocusTag1','Gene1','ProteinName1','GeneOntology1']).to_csv('dataframe1.csv')

ls_genes = []
for items in list(sorted_dict_r2.keys())[0:100]:
    ls_genes.extend([ls_locus_tags_filtered[i] for i in list(items)])
with open('ls_genes_OCKOR_predictions.pickle','wb') as handle:
    pickle.dump(ls_genes,handle)

##

