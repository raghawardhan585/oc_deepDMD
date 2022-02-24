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
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import time
import scipy.stats as st

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= False,n_outputs= -1) # Getting the raw RNAseq and OD600 data to the state and output data format
with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)


##

ls_conditions = list(dict_DATA_ORIGINAL.keys()) # condition identifiers
ls_replicates = list(dict_DATA_ORIGINAL[ls_conditions[0]].keys()) # replicate identifiers
ls_replicates.sort()
ngenes = dict_DATA_ORIGINAL[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].shape[0] # number of genes
ls_locus_tags = list(dict_DATA_ORIGINAL[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].index) # gene list
dict_time_pts = {} # number of time points for each condition
for items in ls_conditions:
    dict_time_pts[items] = dict_DATA_ORIGINAL[items][ls_replicates[0]]['df_X_TPM'].shape[1]

## Plot 1 - mean vs variance plot across time, across genes, across conditons
np_x = np.empty(shape =(0))
np_y = np.empty(shape =(0))
for items in ls_conditions:
    # Generate a 3D numpy matrix with the replicates across the 3rd dimension
    np_cond = np.empty(shape=(ngenes,dict_time_pts[items],0))
    for i in ls_replicates:
        np_cond = np.concatenate([np_cond, np.array(dict_DATA_ORIGINAL[items][i]['df_X_TPM']).reshape(ngenes,-1,1)],axis=2)
    np_cond_mean = np.mean(np_cond,axis=2)
    np_cond_var = np.var(np_cond, axis=2)
    np_x = np.concatenate([np_x, np_cond_mean.reshape(-1)])
    np_y = np.concatenate([np_y, np_cond_var.reshape(-1)])

plt.plot(np_x,np_y,'.')
plt.plot(np_x,np_x+np_x**2/10,'.')
plt.show()

## Distribution of each gene
# SIGNIFICANCE_PERCENT = 0.0025
SIGNIFICANCE_PERCENT = 95
np_all = np.empty(shape=(ngenes,0))
for items,i in itertools.product(ls_conditions,ls_replicates):
    np_all = np.concatenate([np_all,np.array(dict_DATA_ORIGINAL[items][i]['df_X_TPM'])],axis=1)
plt.figure()
count = 0
temp_list = list(range(ngenes))
random.shuffle(temp_list)
# for i in temp_list:
#     # np_temp = np_all[i, :] - np.mean(np_all[i, :])
#     plt.hist(np_all[i,:] - np.mean(np_all[i,:]),10)
#     # plt.title('Gene '+ str(i))
#     count = count + 1
#     if np.mod(count,10)==0:
#         plt.show()
#         time.sleep(1)
#         plt.figure()
#     if count >1000:
#         break
# plt.figure()
# for i in temp_list:
#     # np_temp = np_all[i, :] - np.mean(np_all[i, :])
#     plt.hist(np_all[i,:] - np.mean(np_all[i,:]),10,color='#BABABA')
#     # plt.title('Gene '+ str(i))
# plt.show()

std_all = []
for i in range(ngenes):
    std_all.append(np.std(np_all[i,:],ddof=1))
plt.figure()
plt.hist(np.log(std_all),100)
plt.show()
##
z_val = st.norm.ppf(1 - SIGNIFICANCE_PERCENT/100/2)
count = 0
ls_locus_tags_filtered = []
for i in range(ngenes):
    np_temp = np_all[i,:] - np.mean(np_all[i,:]) # removing the flat constant line across time and across conditions
    gene_mean_val = np.mean(np_temp)
    gene_mean_std = np.std(np_temp, ddof=1)
    lower_bound = gene_mean_val - z_val * gene_mean_std
    upper_bound = gene_mean_val + z_val * gene_mean_std
    if not((0 > lower_bound) and (0 < upper_bound)):
        count = count +1
        ls_locus_tags_filtered.append(ls_locus_tags[i])

print('Number of non constant genes :', count)
df_all = pd.DataFrame(np_all, index=ls_locus_tags)
df_filt = df_all.loc[ls_locus_tags_filtered,:]

dict_DATA_FILTERED = {}
for cond in ls_conditions:
    dict_DATA_FILTERED[cond] = {}
    for rep in ls_replicates:
        dict_DATA_FILTERED[cond][rep] = {}
        dict_DATA_FILTERED[cond][rep]['df_X_TPM'] = copy.deepcopy(dict_DATA_ORIGINAL[cond][rep]['df_X_TPM'].loc[ls_locus_tags_filtered,:])
        dict_DATA_FILTERED[cond][rep]['Y0'] = copy.deepcopy(dict_DATA_ORIGINAL[cond][rep]['Y0'])
        dict_DATA_FILTERED[cond][rep]['Y'] = copy.deepcopy(dict_DATA_ORIGINAL[cond][rep]['Y'])
        try:
            dict_DATA_FILTERED[cond][rep]['U'] = copy.deepcopy(dict_DATA_ORIGINAL[cond][rep]['U'])
        except:
            print('No input detected')
ngenes_filtered = len(ls_locus_tags_filtered)
# TODO Plot to  as early response genes, late response genes and oscillatory



## Data preprocessing - Standardization
SYSTEM_NO = 1000
rnaf.formulate_and_save_Koopman_Data(dict_DATA_FILTERED, ALL_CONDITIONS=ls_conditions, SYSTEM_NO=SYSTEM_NO)
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'

dict_temp = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ls_conditions)

XpTs = dict_temp['train']['XpTs']
UpTs = dict_temp['train']['UpTs']
XfTs = dict_temp['train']['XfTs']
YpTs = dict_temp['train']['YpTs']
YfTs = dict_temp['train']['YfTs']

## Model 1 - Train without U
model_X1 = LinearRegression(fit_intercept=False)
model_X1.fit(XpTs,XfTs)
print('The r^2 value of model 1 trained without inputs:', model_X1.score(XpTs,XfTs))
print('The [A] matrix:')
print(model_X1.coef_)

# n-step predictions
# Predict on all the datasets
XfTs_final_true = np.empty(shape=(0,ngenes_filtered))
XfTs_final_est = np.empty(shape=(0,ngenes_filtered))
for COND,rep in itertools.product(ls_conditions,dict_temp['train']['indices']):
    # Predict the Xf - n step
    XfTsn_hat = dict_temp['scaled'][cond][rep]['XpT'][0:1,:]
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XfTsn_hat = np.concatenate([XfTsn_hat,model_X1.predict(XfTsn_hat[-1:])],axis=0)
    XfTsn_hat = XfTsn_hat[1:]
    XfTs_final_true = np.concatenate([XfTs_final_true,dict_temp['scaled'][cond][rep]['XfT'][-1:]],axis=0)
    XfTs_final_est = np.concatenate([XfTs_final_est, XfTsn_hat[-1:]], axis=0)
    # # Reverse the Xfs
    # XfT_hat = X_scaler.inverse_transform(XfTs_hat)
    # XfTn_hat = X_scaler.inverse_transform(XfTsn_hat)
plt.plot(XfTs_final_true[0:16].reshape(-1),XfTs_final_est[0:16].reshape(-1),'.')
# plt.plot(XfTs_final_true[16:32].reshape(-1),XfTs_final_est[16:32].reshape(-1),'.')
# plt.plot(XfTs_final_true[32:].reshape(-1),XfTs_final_est[32:].reshape(-1),'.')
plt.show()
print('Error in n-step prediction :', r2_score(XfTs_final_true[0:16].reshape(-1),XfTs_final_est[0:16].reshape(-1)))

## Model 2 - Train with U
model_X2 = LinearRegression(fit_intercept=False)
model_X2.fit(np.concatenate([XpTs,UpTs],axis=1),XfTs)
print('The r^2 value:', model_X2.score(np.concatenate([XpTs,UpTs],axis=1),XfTs))
print('The [A b1 b2] matrix:')
print(model_X2.coef_)

# n-step predictions
# Predict on all the datasets
XfTs_final_true = np.empty(shape=(0,ngenes_filtered))
XfTs_final_est = np.empty(shape=(0,ngenes_filtered))
for cond,rep in itertools.product(ls_conditions,dict_temp['train']['indices']):
    # Predict the Xf - n step
    XfTsn_hat = dict_temp['scaled'][cond][rep]['XpT'][0:1,:]
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XfTsn_hat = np.concatenate([XfTsn_hat,model_X2.predict(np.concatenate([XfTsn_hat[-1:],dict_temp['scaled'][cond][rep]['UpT'][-1:]],axis=1))],axis=0)
    XfTsn_hat = XfTsn_hat[1:]
    XfTs_final_true = np.concatenate([XfTs_final_true, dict_temp['scaled'][cond][rep]['XfT'][-1:]],axis=0)
    XfTs_final_est = np.concatenate([XfTs_final_est, XfTsn_hat[-1:]], axis=0)
plt.plot(XfTs_final_true[0:2].reshape(-1),XfTs_final_est[0:2].reshape(-1),'.')
# plt.plot(XfTs_final_true[16:32].reshape(-1),XfTs_final_est[16:32].reshape(-1),'.')
# plt.plot(XfTs_final_true[32:].reshape(-1),XfTs_final_est[32:].reshape(-1),'.')
plt.show()
print('Error in n-step prediction :', r2_score(XfTs_final_true[0:16].reshape(-1),XfTs_final_est[0:16].reshape(-1)))

##
U,s,VT = np.linalg.svd(np.concatenate([XpTs,UpTs],axis=1).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
 #np.min(XpTs.shape)
for r in range(33,34,1): #opt r = 72,33
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Ahat = XfTs.T @ Vr @ np.linalg.inv(Sr) @ UrT
    sb.heatmap(Ahat,cmap='RdBu')
    plt.show()

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
    print('r = ',r,' | r^2 val = ',r2_score(XfT_true.reshape(-1), XfT_est.reshape(-1)))

## n-step predictions
# Predict on all the datasets
dict_DATA_PREDICTED = {'MX':{},'MN':{},'NC':{}}
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
plt.plot(XfTs_final_true[0:16].reshape(-1),XfTs_final_est[0:16].reshape(-1),'.')
plt.plot(XfTs_final_true[16:32].reshape(-1),XfTs_final_est[16:32].reshape(-1),'.')
plt.plot(XfTs_final_true[32:].reshape(-1),XfTs_final_est[32:].reshape(-1),'.')
plt.show()
print('Error in n-step prediction :', r2_score(XfTs_final_true.reshape(-1),XfTs_final_est.reshape(-1)))

##
plt.figure(figsize=(12,12))
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for rep in ls_replicates:
    if rep == 0:
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_FILTERED['MX'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[0], alpha = 0.2, label = 'NC vs MX [data]')
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_FILTERED['MN'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[1], alpha = 0.2, label = 'NC vs MN [data]')
    else:
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:, 7]), np.array(dict_DATA_FILTERED['MX'][rep]['df_X_TPM'].loc[:, 7]), '.', color=ls_colors[0], alpha=0.2)
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:, 7]), np.array(dict_DATA_FILTERED['MN'][rep]['df_X_TPM'].loc[:, 7]), '.', color=ls_colors[1], alpha=0.2)

# plt.plot([-100,30000],[-100,30000],color = ls_colors[2])
# plt.show()

# ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for rep in ls_replicates:
    if rep == 0:
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MX'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[0], label = 'NC vs MX [estimate]')
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MN'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[1], label = 'NC vs MN [estimate]')
    else:
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MX'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[0])
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MN'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[1])
plt.plot([-3000,40000],[-3000,40000],color = ls_colors[2])
plt.legend(ncol=2)
plt.xlim([-3000,40000])
plt.ylim([-3000,40000])
plt.show()



## Fitting the output Y

U,s,VT = np.linalg.svd(np.concatenate([XpTs,XfTs],axis=0).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
Y_scaler = dict_temp['Y_scaler']
 #np.min(XpTs.shape)
for r in range(24,25,1): #opt r = 33
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Chat = np.concatenate([YpTs,YfTs],axis=0).T @ Vr @ np.linalg.inv(Sr) @ UrT
    sb.heatmap(Chat,cmap='RdBu')
    plt.show()

    YT_true = np.empty(shape=(0,20))
    YT_est = np.empty(shape=(0,20))
    for cond,rep in itertools.product(ls_conditions,dict_temp['valid']['indices']):
        XTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['XpT'][0:1],dict_temp['scaled'][cond][rep]['XfT']],axis=0)
        YTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
        YTs_hat = XTs_all @ Chat.T
        YT_true = np.concatenate([YT_true, Y_scaler.inverse_transform(YTs_all)], axis=0)
        YT_est = np.concatenate([YT_est, Y_scaler.inverse_transform(YTs_hat)], axis=0)
    print('r = ',r,' | r^2 val = ',r2_score(YT_true.reshape(-1), YT_est.reshape(-1)))

## Predict full curve
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for cond, rep in itertools.product(ls_conditions, ls_replicates):
    # Predict Xf - n steps
    XTsn_hat = dict_temp['scaled'][cond][rep]['XpT'][0:1,:]
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
        # XTsn_hat = np.concatenate([XTsn_hat, dict_temp['scaled'][cond][rep]['XfT'][j:j+1,:]], axis=0)
    YTsn_hat = XTsn_hat @ Chat.T
    XTn_hat = X_scaler.inverse_transform(XTsn_hat)
    YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
    YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
    YT_true = Y_scaler.inverse_transform(YTs_true)
    plt.plot(YT_true.reshape(-1), '.',color = '#DEDEDE')
    if cond =='MX':
        col_ind =0
    elif cond == 'MN':
        col_ind = 1
    else:
        col_ind = 2
    plt.plot(YTn_hat.reshape(-1), color=ls_colors[col_ind])
plt.show()


##
ls_gene_select_all = list(itertools.combinations(np.arange(len(ls_locus_tags_filtered)),3))
import math
total_choices = math.factorial(len(ls_locus_tags_filtered))/math.factorial(len(ls_locus_tags_filtered)-3)/math.factorial(3)

##
count = 0
rep = dict_temp['test']['indices'][0]
cond = 'MX'
YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
YT_true = Y_scaler.inverse_transform(YTs_true)
XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
    XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
YTsn_hat = XTsn_hat @ Chat.T
YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
dict_gene_select = {}
dict_gene_select['NULL'] = r2_score(YT_true, YTn_hat)

random.shuffle(ls_gene_select_all)

for ls_gene_select in ls_gene_select_all:
    count = count + 1
    XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
    for i in ls_gene_select:
        XTsn_hat[0,i] = 0
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
        for i in ls_gene_select:
            XTsn_hat[-1,i] = 0
    YTsn_hat = XTsn_hat @ Chat.T
    YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
    dict_gene_select[ls_gene_select] = r2_score(YT_true,YTn_hat)
    if np.mod(count,100000) ==0:
        print (count,'/',total_choices,' complete')
        break


dict_gene_select

print(np.min(list(dict_gene_select.values())))


##

plt.hist(dict_gene_select.values(),1000)
plt.xlabel('$r^2$')
plt.ylabel('# 3-gene combinations')
plt.show()
##
import operator
dict_backup1 = copy.deepcopy(dict_gene_select)
sorted_d = dict(sorted(dict_backup1.items(), key=operator.itemgetter(1)))
##
ls_gene_select = list((69, 117, 156))

XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
for i in ls_gene_select:
    XTsn_hat[0, i] = 0
for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
    XTsn_hat = np.concatenate(
        [XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],
        axis=0)
    for i in ls_gene_select:
        XTsn_hat[-1, i] = 0
YTsn_hat = XTsn_hat @ Chat.T
YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
print(r2_score(YT_true, YTn_hat))
##
plt.plot(YT_true.reshape(-1),'.')
plt.plot(YTn_hat.reshape(-1))
plt.show()


##
# import statsmodels as st
# ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
#
# for c,i in itertools.product(ls_conditions,ls_replicates):
#     if c =='MX':
#         col_index = 0
#     elif c == 'MN':
#         col_index = 1
#     else:
#         col_index = 2
#     markerline, stemlines, baseline = plt.stem(st.tsa.stattools.acf(np.array(dict_DATA_ORIGINAL[c][i]['df_X_TPM'].iloc[10, :])), markerfmt='o')
#     plt.setp(markerline, 'color', ls_colors[col_index])
#     plt.setp(stemlines, 'color', ls_colors[col_index])
#     plt.setp(stemlines, 'linestyle', 'dotted')
# plt.show()

# =========================================
## Autocorrelation matrix

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
    # count =0
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
    print('No. of genes thrown out by flatness test (autocorrelation of residuals): ',n_rejects)
    dict_out = copy.deepcopy(dict_data_in)
    for cond, rep in itertools.product(ls_conditions, ls_replicates):
        dict_out[cond][rep]['df_X_TPM'] = dict_data_in[cond][rep]['df_X_TPM'].loc[ls_filtered_genes,:]
    return dict_out

dict_DATA_FILTERED = filter_genes_by_flatness_test(dict_DATA_ORIGINAL)

ls_conditions = list(dict_DATA_ORIGINAL.keys())  # condition identifiers
ls_replicates = list(dict_DATA_ORIGINAL[ls_conditions[0]].keys())  # replicate identifiers
ls_replicates.sort()
ngenes = dict_DATA_ORIGINAL[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].shape[0]  # number of genes
ls_genes = list(dict_DATA_ORIGINAL[ls_conditions[0]][ls_replicates[0]]['df_X_TPM'].index)  # gene list

SYSTEM_NO = 1000
rnaf.formulate_and_save_Koopman_Data(dict_DATA_FILTERED, ALL_CONDITIONS=ls_conditions, SYSTEM_NO=SYSTEM_NO)
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'

dict_temp = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ls_conditions)

XpTs = dict_temp['train']['XpTs']
UpTs = dict_temp['train']['UpTs']
XfTs = dict_temp['train']['XfTs']
YpTs = dict_temp['train']['YpTs']
YfTs = dict_temp['train']['YfTs']
ngenes_filtered = dict_DATA_FILTERED['MX'][0]['df_X_TPM'].shape[0]
ls_locus_tags_filtered = list(dict_DATA_FILTERED['MX'][0]['df_X_TPM'].index)

##
U,s,VT = np.linalg.svd(np.concatenate([XpTs,UpTs],axis=1).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
 #np.min(XpTs.shape)
for r in range(33,34,1): #opt r = 72,33
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Ahat = XfTs.T @ Vr @ np.linalg.inv(Sr) @ UrT
    sb.heatmap(Ahat,cmap='RdBu')
    plt.show()

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
    print('r = ',r,' | r^2 val = ',r2_score(XfT_true.reshape(-1), XfT_est.reshape(-1)))

## n-step predictions
# Predict on all the datasets
dict_DATA_PREDICTED = {'MX':{},'MN':{},'NC':{}}
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
plt.plot(XfTs_final_true[0:16].reshape(-1),XfTs_final_est[0:16].reshape(-1),'.')
plt.plot(XfTs_final_true[16:32].reshape(-1),XfTs_final_est[16:32].reshape(-1),'.')
plt.plot(XfTs_final_true[32:].reshape(-1),XfTs_final_est[32:].reshape(-1),'.')
plt.show()
print('Error in n-step prediction :', r2_score(XfTs_final_true.reshape(-1),XfTs_final_est.reshape(-1)))

##
plt.figure(figsize=(12,12))
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for rep in ls_replicates:
    if rep == 0:
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_FILTERED['MX'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[0], alpha = 0.2, label = 'NC vs MX [data]')
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_FILTERED['MN'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[1], alpha = 0.2, label = 'NC vs MN [data]')
    else:
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:, 7]), np.array(dict_DATA_FILTERED['MX'][rep]['df_X_TPM'].loc[:, 7]), '.', color=ls_colors[0], alpha=0.2)
        plt.plot(np.array(dict_DATA_FILTERED['NC'][rep]['df_X_TPM'].loc[:, 7]), np.array(dict_DATA_FILTERED['MN'][rep]['df_X_TPM'].loc[:, 7]), '.', color=ls_colors[1], alpha=0.2)

# plt.plot([-100,30000],[-100,30000],color = ls_colors[2])
# plt.show()

# ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for rep in ls_replicates:
    if rep == 0:
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MX'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[0], label = 'NC vs MX [estimate]')
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MN'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[1], label = 'NC vs MN [estimate]')
    else:
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MX'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[0])
        plt.plot(np.array(dict_DATA_PREDICTED['NC'][rep]['df_X_TPM'].loc[:,7]),np.array(dict_DATA_PREDICTED['MN'][rep]['df_X_TPM'].loc[:,7]),'.',color = ls_colors[1])
plt.plot([-3000,40000],[-3000,40000],color = ls_colors[2])
plt.legend(ncol=2)
plt.xlim([-3000,40000])
plt.ylim([-3000,40000])
plt.show()



## Fitting the output Y

U,s,VT = np.linalg.svd(np.concatenate([XpTs,XfTs],axis=0).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
Y_scaler = dict_temp['Y_scaler']
 #np.min(XpTs.shape)
for r in range(24,25,1): #opt r = 33
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Chat = np.concatenate([YpTs,YfTs],axis=0).T @ Vr @ np.linalg.inv(Sr) @ UrT
    sb.heatmap(Chat,cmap='RdBu')
    plt.show()

    YT_true = np.empty(shape=(0,20))
    YT_est = np.empty(shape=(0,20))
    for cond,rep in itertools.product(ls_conditions,dict_temp['valid']['indices']):
        XTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['XpT'][0:1],dict_temp['scaled'][cond][rep]['XfT']],axis=0)
        YTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
        YTs_hat = XTs_all @ Chat.T
        YT_true = np.concatenate([YT_true, Y_scaler.inverse_transform(YTs_all)], axis=0)
        YT_est = np.concatenate([YT_est, Y_scaler.inverse_transform(YTs_hat)], axis=0)
    print('r = ',r,' | r^2 val = ',r2_score(YT_true.reshape(-1), YT_est.reshape(-1)))

## Predict full curve
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
for cond, rep in itertools.product(ls_conditions, ls_replicates):
    # Predict Xf - n steps
    XTsn_hat = dict_temp['scaled'][cond][rep]['XpT'][0:1,:]
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
        # XTsn_hat = np.concatenate([XTsn_hat, dict_temp['scaled'][cond][rep]['XfT'][j:j+1,:]], axis=0)
    YTsn_hat = XTsn_hat @ Chat.T
    XTn_hat = X_scaler.inverse_transform(XTsn_hat)
    YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
    YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
    YT_true = Y_scaler.inverse_transform(YTs_true)
    plt.plot(YT_true.reshape(-1), '.',color = '#DEDEDE')
    if cond =='MX':
        col_ind =0
    elif cond == 'MN':
        col_ind = 1
    else:
        col_ind = 2
    plt.plot(YTn_hat.reshape(-1), color=ls_colors[col_ind])
plt.show()

##
ls_filtered_gene_tags  = np.arange(ngenes_filtered)
set_filtered_gene_tags = set(ls_filtered_gene_tags)


rep = dict_temp['test']['indices'][0]
cond = 'MN'
YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
YT_true = Y_scaler.inverse_transform(YTs_true)
XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
    XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
YTsn_hat = XTsn_hat @ Chat.T
YTn_hat0 = Y_scaler.inverse_transform(YTsn_hat)

dict_gene_select = {}
# dict_gene_select['NULL'] = r2_score(YT_true, YTn_hat)

# for ls_gene_select in ls_gene_select_all:
for count in range(1000):
    ls_gene_select = random.sample(set_filtered_gene_tags,3)
    XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
    for i in ls_gene_select:
        XTsn_hat[0,i] = 0
    for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
        XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
        for i in ls_gene_select:
            XTsn_hat[-1,i] = 0
    YTsn_hat = XTsn_hat @ Chat.T
    YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
    # dict_gene_select[ls_gene_select] = r2_score(YT_true,YTn_hat)
    dict_gene_select[tuple(ls_gene_select)] = r2_score(YTn_hat0, YTn_hat)
    # if np.mod(count,100000) ==0:
    #     # print (count,'/',total_choices,' complete')
    #     break


dict_gene_select

print(np.min(list(dict_gene_select.values())))

plt.hist(dict_gene_select.values(), bins = 50)
plt.show()
