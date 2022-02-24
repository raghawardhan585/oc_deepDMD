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
import operator
import seaborn as sns

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= False,n_outputs= 40) # Getting the raw RNAseq and OD600 data to the state and output data format
# rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= True,n_outputs= -1)
with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)
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
dict_DATA_FILTERED = filter_genes_by_flatness_test(dict_DATA_ORIGINAL, significance_level_percent= 5)
ngenes_filtered = dict_DATA_FILTERED['MX'][0]['df_X_TPM'].shape[0]
ls_locus_tags_filtered = list(dict_DATA_FILTERED['MX'][0]['df_X_TPM'].index)
print('Number of filtered genes: ', ngenes_filtered)

# Genes in comparison to DESeq2
df_DE = pd.read_csv('/Users/shara/Desktop/masigpro_RNAseq/DESEq_ordered_by_adjusted_p_values.csv',index_col=0)
df_DE_filt = df_DE[df_DE['padj']<1e-15]
ls_DEfiltered_genes = list(df_DE_filt.index)
print('No of genes filtered by autocorrelation: ', ngenes_filtered)
print('No of genes filtered by DESeq2: ', df_DE_filt.shape[0])
print('No of intersecting genes: ',len(set(ls_locus_tags_filtered) & set(df_DE_filt.index)))
dict_DATA_FILTERED = copy.deepcopy(dict_DATA_ORIGINAL)
for cond, rep in itertools.product(ls_conditions, ls_replicates):
    dict_DATA_FILTERED[cond][rep]['df_X_TPM'] = dict_DATA_FILTERED[cond][rep]['df_X_TPM'].loc[ls_DEfiltered_genes, :]
ngenes_filtered = dict_DATA_FILTERED['MX'][0]['df_X_TPM'].shape[0]
ls_locus_tags_filtered = list(dict_DATA_FILTERED['MX'][0]['df_X_TPM'].index)

#
# def gene_heat_map(dict_GrowthCurve,THIS_COND,THIS_REPLICATE,params):
#     df_sb_DATA = copy.deepcopy(dict_GrowthCurve[THIS_COND][THIS_REPLICATE]['df_X_TPM'].loc[params['Y label'],params['All timepoints']])
#     df_sb_DATA = np.log10(df_sb_DATA+1)
#     df_sb_DATA = df_sb_DATA.replace([np.inf, -np.inf], np.nan)
#     if params['Normalize per gene']:
#         df_sb_DATA = (df_sb_DATA - df_sb_DATA.min().min()) / (df_sb_DATA.max().max() - df_sb_DATA.min().min())
#     if params['Normalize per gene']:
#         This_Center = 0.5;
#     else:
#         This_Center = df_sb_DATA.mean().mean()  # (5.0-0.5)/2;#np.mean(nd_sb_DATA);
#
#     xlabel = [str(i) + ' hr' for i in params['All timepoints']]
#     # xlabel = ['10 min','50 min','70 min','90 min','110 min','120 min']
#     N_plots_max = int(np.ceil(len(params['Y label']) / params['No of genes per plot']))
#     if params['No of plots'] == -1:
#         N_plots = N_plots_max
#     else:
#         if params['No of plots']> N_plots_max:
#             N_plots = N_plots_max
#         else:
#             N_plots = params['No of plots']
#
#     if type(params['figsize']) is not tuple:
#         params['figsize'] = (50 * N_plots, np.int(params['No of genes per plot'] *2)) #/6
#     f, ax_plot = plt.subplots(1, N_plots + 1, figsize=(params['figsize']),gridspec_kw={'width_ratios': list(np.append(np.ones(shape=N_plots), 0.08))})
#     for i in range(N_plots):
#         N_start = i * params['No of genes per plot']
#         N_end = np.min([N_start + params['No of genes per plot'],len(params['Y label'])])
#         ylabel = params['Y label'][N_start:N_end]
#         ylabel = [elem[0:-4] for elem in ylabel]
#         if i == N_plots - 1:
#             g_i = sb.heatmap(df_sb_DATA.iloc[N_start:N_end, :], xticklabels=xlabel, center=This_Center,
#                              cmap=sb.diverging_palette(240, 11.75, s=99, l=30.2, n=15), linewidth=1, square=False,
#                              vmin=-0.5, vmax=5.0, ax=ax_plot[i], cbar_ax=ax_plot[i + 1],cbar_kws={"shrink": params['Color Bar Size']})
#             cmap_labels = ax_plot[N_plots].axes.get_yticks()
#             ax_plot[N_plots].axes.set_yticklabels(cmap_labels, params['legend tick label param'])
#         else:
#             # ax = sb.heatmap(nd_sb_DATA,xticklabels = xlabel,center=This_Center ,cmap=sb.diverging_palette(240, 11.75,s=99,l=30.2, n=15),linewidth=1,square=False,cbar=True)
#             g_i = sb.heatmap(df_sb_DATA.iloc[N_start:N_end, :], xticklabels=xlabel, center=This_Center,
#                              cmap=sb.diverging_palette(240, 11.75, s=99, l=30.2, n=15), linewidth=1, square=False,
#                              vmin=-0.5, vmax=5.0, cbar=False, ax=ax_plot[i])
#         b, t = g_i.axes.get_ylim()  # discover the values for bottom and top
#         b += 0.5  # Add 0.5 to the bottom
#         t -= 0.5  # Subtract 0.5 from the top
#         g_i.axes.set_ylim(b, t)  # update the ylim(bottom, top) values
#         g_i.axes.set_xticklabels(xlabel, params['x tick label param'], rotation = 0)
#         g_i.axes.set_yticklabels(ylabel, params['y tick label param'], rotation = 0)
#         # if THIS_COND == 'MAX':
#         #    g_i.axes.set_yticklabels([])
#     return f,ax_plot
#
#
# # Graph Input Parameters
# GENE_SORT_CONDITION = 'MAX'
# gene_heat_map_params = {}
# gene_heat_map_params['No of genes per plot'] = 150;
# gene_heat_map_params['No of plots'] = 1 # if -1 , it finds out the number of plots required
# gene_heat_map_params['Normalize per gene'] = False
# gene_heat_map_params['All timepoints'] = [3,4,5,6,7]
# gene_heat_map_params['Y label'] = ls_locus_tags_filtered
# gene_heat_map_params['figsize'] = (80,100)#-1
# # gene_heat_map_params['figsize'] = (50 * N_plots, np.int(params['No of genes per plot'] / 6))
# gene_heat_map_params['x tick label param'] = {'fontsize':202} #128
# gene_heat_map_params['y tick label param'] = {'fontsize':24} #128
# gene_heat_map_params['legend tick label param'] = {'fontsize':102} #128
# gene_heat_map_params['Color Bar Size'] = 15
# THIS_REPLICATE = 7#ALL_REPLICATES[0:16]
# f,ax = gene_heat_map(dict_DATA_FILTERED,'NC',7,gene_heat_map_params)
# f.savefig('Plots/DATA_VISUALIZATION_MIN_orderedbyMIN.png')

#
# DS = 4
# time = np.arange(60,60*8,3)/60
# plt.figure(figsize=(7,6))
# plt.plot(time[0::DS],np.array(dict_DATA_ORIGINAL['MX'][7]['Y']).T.reshape(-1)[0::DS],'.',label = 'MAX growth',linewidth =7)
# plt.plot(time[0::DS],np.array(dict_DATA_ORIGINAL['MN'][7]['Y']).T.reshape(-1)[0::DS],'.',label = 'MIN growth',linewidth =7)
# plt.plot(time[0::DS],np.array(dict_DATA_ORIGINAL['NC'][7]['Y']).T.reshape(-1)[0::DS],'.',label = 'NC growth',linewidth =7)
# plt.legend(ncol=2,loc = 'upper center', fontsize= 20)
# plt.ylim([-1,5])
# plt.xticks([1,2,3,4,5,6,7,8])
# plt.xlabel('Time (hrs)')
# plt.ylabel('Fitness')
# plt.savefig('Plots/DATA_VISUALIZATION_MIN_orderedbyMIN.png')
##
# for cond, rep in itertools.product(ls_conditions, ls_replicates):
#     plt.plot(np.array(dict_DATA_FILTERED[cond][rep]['df_X_TPM']).T,color = '#BABABA')
# plt.show()
##


SYSTEM_NO = 1000
rnaf.formulate_and_save_Koopman_Data(dict_DATA_FILTERED, ALL_CONDITIONS=ls_conditions, SYSTEM_NO=SYSTEM_NO)
ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'

##
# ls_conditions = ['MX']
dict_temp = rnaf.get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ls_conditions)


XpTs = dict_temp['train']['XpTs']
UpTs = dict_temp['train']['UpTs']
XfTs = dict_temp['train']['XfTs']
YpTs = dict_temp['train']['YpTs']
YfTs = dict_temp['train']['YfTs']


##
U,s,VT = np.linalg.svd(np.concatenate([XpTs,UpTs],axis=1).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
 #np.min(XpTs.shape)
# for r in range(10,90,5): #opt r = 72,33
for r in range(31, 32, 1):  # opt r = 72,33
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
plt.plot([-3000,20000],[-3000,20000],color = ls_colors[2])
plt.legend(ncol=2)
plt.xlim([-3000,20000])
plt.ylim([-3000,20000])
plt.show()



## Fitting the output Y

U,s,VT = np.linalg.svd(np.concatenate([XpTs,XfTs],axis=0).T)
# U,s,VT = np.linalg.svd(XpTs.T)
X_scaler = dict_temp['X_scaler']
U_scaler = dict_temp['U_scaler']
Y_scaler = dict_temp['Y_scaler']
 #np.min(XpTs.shape)
# for r in range(10,50,5): #opt r = 33
for r in range(1, 30, 1):  # opt r = 33
    Ur = U[:,0:r]
    UrT = np.conj(Ur.T)
    Sr = np.diag(s[0:r])
    V = np.conj(VT.T)
    Vr = V[:,0:r]
    Chat = np.concatenate([YpTs,YfTs],axis=0).T @ Vr @ np.linalg.inv(Sr) @ UrT
    # sb.heatmap(Chat,cmap='RdBu')
    # plt.show()

    YT_true = np.empty(shape=(0,20))
    YT_est = np.empty(shape=(0,20))
    for cond,rep in itertools.product(ls_conditions,dict_temp['train']['indices']):
        XTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['XpT'][0:1],dict_temp['scaled'][cond][rep]['XfT']],axis=0)
        YTs_all = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'][0:1], dict_temp['scaled'][cond][rep]['YfT']],axis=0)
        YTs_hat = XTs_all @ Chat.T
        YT_true = np.concatenate([YT_true, Y_scaler.inverse_transform(YTs_all)], axis=0)
        YT_est = np.concatenate([YT_est, Y_scaler.inverse_transform(YTs_hat)], axis=0)
    print('r = ',r,' | Validation Data r^2 val = ',r2_score(YT_true.reshape(-1), YT_est.reshape(-1)))

## C and C0 matrices
Y_sig = np.diag(Y_scaler.var_)
X_sig = np.diag(X_scaler.var_)
U_sig = np.diag(U_scaler.var_)
Y_mu = Y_scaler.mean_.reshape(-1,1)
X_mu = X_scaler.mean_.reshape(-1,1)
U_mu = U_scaler.mean_.reshape(-1,1)

C = np.matmul(np.matmul(Y_sig,Chat),np.linalg.inv(X_sig))
C0 = Y_mu - np.matmul(np.matmul(np.matmul(Y_sig,Chat),np.linalg.inv(X_sig)),X_mu)

sns.heatmap(C)
plt.show()
sns.heatmap(C0)
plt.show()


##
As = Ahat[:,0:283]
Bs = Ahat[:,283:285]
B0 = X_mu - np.matmul(np.matmul(np.matmul(X_sig,As),np.linalg.inv(X_sig)),X_mu) - np.matmul(np.matmul(np.matmul(X_sig,Bs),np.linalg.inv(U_sig)),U_mu)
print('2-norm of B0: ',np.linalg.norm(B0,2))
A = np.matmul(np.matmul(X_sig,As),np.linalg.inv(X_sig))
print('2-norm of A: ',np.linalg.norm(A,2))
B = np.matmul(np.matmul(X_sig,Bs),np.linalg.inv(U_sig))
print('2-norm of B: ',np.linalg.norm(B,2))

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

## ONE CONDITION
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
# dict_gene_select['NULL'] = r2_score(YT_true, YTn_hat)
epsilon = 1e-3
# for ls_gene_select in ls_gene_select_all:
# for count in range(100000):
#     try:
#         ls_gene_select = random.sample(set_filtered_gene_nos,1)
#         XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
#         for i in ls_gene_select:
#             XTsn_hat[0,i] = 0
#         for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
#             XTsn_hat = np.concatenate([XTsn_hat, np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]], axis=1) @ Ahat.T],axis=0)
#             for i in ls_gene_select:
#                 XTsn_hat[-1,i] = 0
#         YTsn_hat = XTsn_hat @ Chat.T
#         YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
#         # dict_gene_select[ls_gene_select] = r2_score(YT_true,YTn_hat)
#         dict_gene_select_r2[tuple(ls_gene_select)] = r2_score(YTn_hat0, YTn_hat)
#         dict_gene_select_fc[tuple(ls_gene_select)] = np.median(np.abs(1-((YTn_hat[:,0]+epsilon)/(YTn_hat0[:,0]+epsilon))))
#         # if np.mod(count,100000) ==0:
#         #     # print (count,'/',total_choices,' complete')
#         #     break
#     except:
#         break


# simluate single gene knockouts
ls_gene_select = list(set_filtered_gene_nos)
for i in range(len(ls_gene_select)):
    try:
        XTsn_hat = copy.deepcopy(dict_temp['scaled'][cond][rep]['XpT'][0:1, :])
        XTsn_hat[0, i] = 0
        for j in range(len(dict_temp['scaled'][cond][rep]['XfT'])):
            XTsn_hat = np.concatenate([XTsn_hat,
                                       np.concatenate([XTsn_hat[-1:], dict_temp['scaled'][cond][rep]['UpT'][-1:]],
                                                      axis=1) @ Ahat.T], axis=0)
            XTsn_hat[-1, i] = 0
        YTsn_hat = XTsn_hat @ Chat.T
        YTn_hat = Y_scaler.inverse_transform(YTsn_hat)
        # dict_gene_select[ls_gene_select] = r2_score(YT_true,YTn_hat)
        dict_gene_select_r2[tuple([ls_gene_select[i]])] = r2_score(YTn_hat0, YTn_hat)
        dict_gene_select_fc[tuple([ls_gene_select[i]])] = np.median(
            np.abs(1 - ((YTn_hat[:, 0] + epsilon) / (YTn_hat0[:, 0] + epsilon))))
        # if np.mod(count,100000) ==0:
        #     # print (count,'/',total_choices,' complete')
        #     break
    except:
        break


print(np.min(list(dict_gene_select_r2.values())))
print(np.min(list(dict_gene_select_fc.values())))

plt.hist(dict_gene_select_r2.values(), bins = 50)
plt.xlabel('$r^2$')
plt.ylabel('Frequency of \n 3-gene knockouts')
plt.show()

# plt.hist(dict_gene_select_fc.values(), bins = 50)
# plt.show()

#
sorted_dict_r2 = dict(sorted(dict_gene_select_r2.items(), key=operator.itemgetter(1)))
sorted_dict_fc = dict(sorted(dict_gene_select_fc.items(), reverse=True, key=operator.itemgetter(1)))

## Enoch - validate the gene list by predicting the output curves by generating
# Prediction options
SCALED = False
COMPUTE_OD = False
N_STEP = True
# Plot options
WITH_EMBEDDING = True
PLOT_GRAY = True

N_GENES_TO_PLOT = 200

Y0 = dict_DATA_ORIGINAL[cond][rep]['Y0']
ls_ordered_indices = []
ls_ordered_genes = []
for items in list(sorted_dict_r2.keys())[0:N_GENES_TO_PLOT]:
    ls_ordered_indices.extend([items[0]])
    ls_ordered_genes.extend([ls_locus_tags_filtered[i]  for i in list(items)])

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
        YT_true = 2 ** YT_true * Y0
        YTn_hat_wild = 2 ** YTn_hat_wild * Y0
else:
    YTsn_hat_wild = XTsn_hat @ Chat.T
    YTs_true = np.concatenate([dict_temp['scaled'][cond][rep]['YpT'], dict_temp['scaled'][cond][rep]['YfT'][-1:,:]],axis=0)


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
        if COMPUTE_OD:
            YTn_hat = 2 ** YTn_hat * Y0
        dist = np.round(np.sqrt(np.sum(np.maximum(0, YTn_hat_wild.reshape(-1) - YTn_hat.reshape(-1)) ** 2)), 2)
        # Plotting based on options
        if WITH_EMBEDDING and PLOT_GRAY:
            plt.plot(ls_time, YTn_hat.reshape(-1), color = 'gray')
        elif WITH_EMBEDDING:
            plt.plot(ls_time, YTn_hat.reshape(-1), label=ls_ordered_genes[i] + '( - ' + str(dist) + ')', linewidth=2)
        elif PLOT_GRAY:
            plt.plot(np.arange(1, 8), YTn_hat[:, 0].reshape(-1), color='gray')
        else:
            plt.plot(np.arange(1, 8), YTn_hat[:, 0], label=ls_ordered_genes[i] + '( - ' + str(dist) + ')', linewidth=2)
    else:
        if COMPUTE_OD:
            print('Wrong stat for scaling. Scaling = False for OD computation')
        dist = np.round(np.sqrt(np.sum(np.maximum(0, YTsn_hat_wild.reshape(-1) - YTsn_hat.reshape(-1)) ** 2)), 2)
        # Plotting based on options
        if WITH_EMBEDDING and PLOT_GRAY:
            plt.plot(ls_time, YTsn_hat.reshape(-1), color = 'gray')
        elif WITH_EMBEDDING:
            plt.plot(ls_time, YTsn_hat.reshape(-1), label=ls_ordered_genes[i] + '( - ' + str(dist) + ')', linewidth=2)
        elif PLOT_GRAY:
            plt.plot(np.arange(1, 8), YTsn_hat[:, 0], color='gray')
        else:
            plt.plot(np.arange(1, 8), YTsn_hat[:, 0], label=ls_ordered_genes[i] + '( - ' + str(dist) + ')', linewidth=2)


if WITH_EMBEDDING and SCALED:
    plt.plot(ls_time, YTs_true.reshape(-1), label='Wild (data)', marker='.', markersize=0)
    plt.plot(ls_time, YTsn_hat_wild.reshape(-1), label='Wild (OCKOR)', marker='.', markersize=0)
elif WITH_EMBEDDING:
    plt.plot(ls_time, YT_true.reshape(-1), label='Wild (data)', marker='.', markersize=0)
    plt.plot(ls_time, YTn_hat_wild.reshape(-1), label='Wild (OCKOR)', marker='.', markersize=0)
elif SCALED:
    plt.plot(np.arange(1, 8), YTs_true[:, 0], label='Wild (data)', marker='.', markersize=0)
    plt.plot(np.arange(1, 8), YTsn_hat_wild[:, 0], label='Wild (OCKOR)', marker='.', markersize=0)
else:
    plt.plot(np.arange(1, 8), YT_true[:, 0], label='Wild (data)', marker='.', markersize=0)
    plt.plot(np.arange(1, 8), YTn_hat_wild[:, 0], label='Wild (OCKOR)', marker='.', markersize=0)



plt.legend(loc = 'lower right',ncol = 2,fontsize = 16)
plt.xlabel('Time (hrs)')
plt.ylabel('$OD_{600}$')
# plt.title('Scaled output')
# plt.title('')
plt.xlim([0,7.1])
if COMPUTE_OD:
    plt.ylim(-2,1.5)
else:
    plt.ylim(0, 1.5)
plt.show()







## Export the gene list
dict_out = {}
np_data = np.empty(shape=(0,4))
for items in list(sorted_dict_r2.keys())[0:100]:
    ls_genes_i = [ls_locus_tags_filtered[i]  for i in list(items)]
    entry_i = rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags=ls_genes_i,search_columns='genes(OLN),genes(PREFERRED),protein names,go(biological process)')
    np_data = np.concatenate([np_data,np.array(entry_i).reshape(1,-1)],axis=0)

pd.DataFrame(np_data,columns=['LocusTag1','Gene1','ProteinName1','GeneOntology1','LocusTag2','Gene2','ProteinName2','GeneOntology2','LocusTag3','Gene3','ProteinName3','GeneOntology3']).to_csv('dataframe1.csv')

ls_genes = []
for items in list(sorted_dict_r2.keys())[0:100]:
    ls_genes.extend([ls_locus_tags_filtered[i] for i in list(items)])
with open('ls_genes_OCKOR_predictions.pickle','wb') as handle:
    pickle.dump(ls_genes,handle)