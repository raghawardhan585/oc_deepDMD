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
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

##
# To get the RNAseq and OD data to RAW format of X and Y data
rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True)
# rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = False)

## Open the RAW datafile

with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)
# dict_DATA = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA, MEAN_TPM_THRESHOLD = 1, ALL_CONDITIONS= ['MX'])
dict_DATA_max_denoised = copy.deepcopy(dict_DATA_ORIGINAL)
# dict_DATA_max_denoised['MX'] = rnaf.denoise_using_PCA(dict_DATA_max_denoised['MX'], PCA_THRESHOLD = 99, NORMALIZE=True, PLOT_SCREE=False)

ALL_CONDITIONS = ['MX','NC','MN']



## Combining all the three filtering strategies
dict_DATA_max_denoised = copy.deepcopy(dict_DATA_ORIGINAL)
ALL_CONDITIONS = ['MX','NC','MN']
# 1 - Get the genes from gene ontology
dict_growth_genes = rnaf.get_PputidaKT2440_growth_genes()
ls_genes_biocyc = set(dict_growth_genes['cell_cycle']).union(set(dict_growth_genes['cell_division']))
_,ls_genes_uniprot = rnaf.get_Uniprot_cell_division_genes_and_cell_cycle_genes()

ls_genes_uniprot_biocyc = list(set(ls_genes_biocyc).union(set(ls_genes_uniprot)))
dict_data1 = {}
for condition in ALL_CONDITIONS:
    dict_data1[condition] = {}
    for items in dict_DATA_max_denoised[condition].keys():
        dict_data1[condition][items] = {'df_X_TPM': dict_DATA_max_denoised[condition][items]['df_X_TPM'].loc[ls_genes_uniprot_biocyc,:], 'Y0': dict_DATA_max_denoised[condition][items]['Y0'], 'Y': dict_DATA_max_denoised[condition][items]['Y']}

## Plot all 48 curves


ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
n_genes_uniprot_biocyc = len(ls_genes_uniprot_biocyc)
n_rows = 5#np.int(np.ceil(np.sqrt(n_genes_uniprot_biocyc)))
n_cols = np.int(np.ceil(n_genes_uniprot_biocyc/ n_rows))
f,ax = plt.subplots(n_rows, n_cols, figsize =(60,30))
ax_l = ax.reshape(-1)
for i in range(n_genes_uniprot_biocyc):
    df_gene_replicates_MAX = pd.DataFrame([])
    df_gene_replicates_MIN = pd.DataFrame([])
    df_gene_replicates_NC = pd.DataFrame([])
    for j in range(16):
        try:
            df_gene_replicates_MAX = pd.concat([df_gene_replicates_MAX,dict_data1['MX'][j]['df_X_TPM'].iloc[i:i+1,:]],axis=0)
            df_gene_replicates_MIN = pd.concat([df_gene_replicates_MIN, dict_data1['MN'][j]['df_X_TPM'].iloc[i:i + 1, :]], axis=0)
            df_gene_replicates_NC = pd.concat([df_gene_replicates_NC, dict_data1['NC'][j]['df_X_TPM'].iloc[i:i + 1, :]], axis=0)
        except:
            df_gene_replicates_MAX = dict_data1['MX'][j]['df_X_TPM'].iloc[i:i + 1, :]
            df_gene_replicates_MIN = dict_data1['MN'][j]['df_X_TPM'].iloc[i:i + 1, :]
            df_gene_replicates_NC = dict_data1['NC'][j]['df_X_TPM'].iloc[i:i + 1, :]
    ax_l[i].errorbar(np.array([1,2,3,4,5,6,7]), df_gene_replicates_MAX.mean(axis=0),yerr =df_gene_replicates_MAX.std(axis=0),color = ls_colors[0],capsize =8)
    ax_l[i].errorbar(np.array([3, 4, 5, 6, 7]), df_gene_replicates_MIN.mean(axis=0), yerr=df_gene_replicates_MIN.std(axis=0),color = ls_colors[1],capsize =8)
    ax_l[i].errorbar(np.array([3, 4, 5, 6, 7]), df_gene_replicates_NC.mean(axis=0),yerr=df_gene_replicates_NC.std(axis=0), color=ls_colors[2],capsize =8)
    ax_l[i].set_title(ls_genes_uniprot_biocyc[i])

f.show()
## Differential expression score
# 2 -
ALL_CONDITIONS = ['MX']
TIME_POINT = 7
T_START = 1
df_genes_diff_dynamics_MX_MX = pd.DataFrame([])
for i in range(16):
    df_temp = np.abs(np.log2(dict_data1['MX'][i]['df_X_TPM'].loc[:,TIME_POINT].divide(dict_data1['MX'][i]['df_X_TPM'].loc[:,T_START]))).sort_values(ascending=False)
    try:
        df_genes_diff_dynamics_MX_MX = pd.concat([df_genes_diff_dynamics_MX_MX,pd.DataFrame(df_temp,columns=[i])],axis=1)
    except:
        df_genes_diff_dynamics_MX_MX = pd.DataFrame(df_temp,columns=[i])
# print(pd.concat([df_genes_diff_dynamics_MX_MX.mean(axis=1)-df_genes_diff_dynamics_MX_MX.std(axis=1),df_genes_diff_dynamics_MX_MX.mean(axis=1)+df_genes_diff_dynamics_MX_MX.std(axis=1)],axis=1))
print(pd.concat([df_genes_diff_dynamics_MX_MX.min(axis=1),df_genes_diff_dynamics_MX_MX.max(axis=1)],axis=1))

ls_genes2 = []
for i in range(df_genes_diff_dynamics_MX_MX.shape[0]):
    if df_genes_diff_dynamics_MX_MX.iloc[i,:].max() >4:
        ls_genes2.append(df_genes_diff_dynamics_MX_MX.index[i])



ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
n_genes2 = len(ls_genes2)
n_rows = 2#np.int(np.ceil(np.sqrt(n_genes_uniprot_biocyc)))
n_cols = np.int(np.ceil(n_genes2/ n_rows))
f,ax = plt.subplots(n_rows, n_cols, figsize =(50,10))
ax_l = ax.reshape(-1)
for i in range(n_genes2):
    df_gene_replicates_MAX = pd.DataFrame([])
    df_gene_replicates_MIN = pd.DataFrame([])
    df_gene_replicates_NC = pd.DataFrame([])
    gene_name = ls_genes2[i]
    for j in range(16):
        try:
            df_gene_replicates_MAX = pd.concat([df_gene_replicates_MAX,pd.DataFrame(dict_data1['MX'][j]['df_X_TPM'].loc[gene_name,:]).T],axis=0)
            df_gene_replicates_MIN = pd.concat([df_gene_replicates_MIN, pd.DataFrame(dict_data1['MN'][j]['df_X_TPM'].loc[gene_name, :]).T], axis=0)
            df_gene_replicates_NC = pd.concat([df_gene_replicates_NC, pd.DataFrame(dict_data1['NC'][j]['df_X_TPM'].loc[gene_name, :]).T], axis=0)
        except:
            df_gene_replicates_MAX = pd.DataFrame(dict_data1['MX'][j]['df_X_TPM'].loc[gene_name, :]).T
            # df_gene_replicates_MIN = pd.DataFrame(dict_data1['MN'][j]['df_X_TPM'].loc[gene_name, :]).T
            # df_gene_replicates_NC = pd.DataFrame(dict_data1['NC'][j]['df_X_TPM'].loc[gene_name, :]).T
    ax_l[i].errorbar(np.array([1,2,3,4,5,6,7]), df_gene_replicates_MAX.mean(axis=0),yerr =df_gene_replicates_MAX.std(axis=0),color = ls_colors[0],capsize =8)
    ax_l[i].errorbar(np.array([3, 4, 5, 6, 7]), df_gene_replicates_MIN.mean(axis=0), yerr=df_gene_replicates_MIN.std(axis=0),color = ls_colors[1],capsize =8)
    ax_l[i].errorbar(np.array([3, 4, 5, 6, 7]), df_gene_replicates_NC.mean(axis=0),yerr=df_gene_replicates_NC.std(axis=0), color=ls_colors[2],capsize =8)
    ax_l[i].set_title(ls_genes2[i])

f.show()

## Collect the data using

# ALL_CONDITIONS = ['MX','MN','NC']
ALL_CONDITIONS = ['MX']
dict_data = {}
for condition in ALL_CONDITIONS:
    dict_data[condition] = {}
    for items in dict_DATA_max_denoised[condition].keys():
        dict_data[condition][items] = {'df_X_TPM': dict_DATA_max_denoised[condition][items]['df_X_TPM'].loc[ls_genes2,:], 'Y0': dict_DATA_max_denoised[condition][items]['Y0'], 'Y': dict_DATA_max_denoised[condition][items]['Y']}

## Plotting the states as a function of time
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
curve = 0
f,ax = plt.subplots(7,1,sharex=True,figsize=(30,14))
for time_pt,COND_NO in itertools.product(range(1,8),range(len(ALL_CONDITIONS))):
    COND = ALL_CONDITIONS[COND_NO]
    # for curve in range(16):
    # ax[time_pt-1].plot(np.array(dict_DATA_ORIGINAL['MX'][curve]['df_X_TPM'].loc[:, time_pt]))
    # ax[time_pt - 1].plot(np.array(dict_DATA_max_denoised['MX'][curve]['df_X_TPM'].loc[:,time_pt]))
    try:
        # ax[time_pt - 1].plot(np.log10(np.array(dict_data[COND][curve]['df_X_TPM'].loc[:, time_pt])),color = ls_colors[COND_NO])
        ax[time_pt - 1].plot(np.array(dict_data[COND][curve]['df_X_TPM'].loc[:, time_pt]), color=ls_colors[COND_NO])
    except:
        print("Skipping condition ", COND, ' time point ', time_pt)
    # ax[time_pt - 1].set_xlim([120])
for time_pt in range(1, 8):
    # ax[time_pt - 1].set_ylim([0, 40000])
    # ax[time_pt - 1].set_ylim([0, 10000])
    ax[time_pt-1].set_title('Time Point : ' + str(time_pt),fontsize=24)
ax[-1].set_xlabel('Gene Locus Tag')
f.show()

## Sorting the MAX dataset to deepDMD format and doing the train-validation-test split

ls_all_indices = list(dict_data[ALL_CONDITIONS[0]].keys())
random.shuffle(ls_all_indices)
ls_train_indices = ls_all_indices[0:14]
# ls_valid_indices = ls_all_indices[12:14]
ls_test_indices = ls_all_indices[14:16]
n_states = dict_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['df_X_TPM'].shape[0]
n_outputs = dict_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['Y'].shape[0]

dict_DMD_train = {'Xp' : np.empty(shape=(0,n_states)), 'Xf': np.empty(shape=(0,n_states)),'Yp' : np.empty(shape=(0,n_outputs)), 'Yf' : np.empty(shape=(0,n_outputs))}
# for COND,i in itertools.product(ALL_CONDITIONS,ls_train_indices):
for i, COND in itertools.product(ls_train_indices,ALL_CONDITIONS):
    dict_DMD_train['Xp'] = np.concatenate([dict_DMD_train['Xp'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:,0:-1]).T],axis=0)
    dict_DMD_train['Xf'] = np.concatenate([dict_DMD_train['Xf'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
    dict_DMD_train['Yp'] = np.concatenate([dict_DMD_train['Yp'], np.array(dict_data[COND][i]['Y'].iloc[:, 0:-1]).T], axis=0)
    dict_DMD_train['Yf'] = np.concatenate([dict_DMD_train['Yf'], np.array(dict_data[COND][i]['Y'].iloc[:, 1:]).T], axis=0)

# dict_DMD_valid = {'Xp' : np.empty(shape=(0,n_states)), 'Xf': np.empty(shape=(0,n_states)),'Yp' : np.empty(shape=(0,n_outputs)), 'Yf' : np.empty(shape=(0,n_outputs))}
# for i in ls_valid_indices:
#     dict_DMD_valid['Xp'] = np.concatenate([dict_DMD_valid['Xp'], np.array(dict_MAX[i]['df_X_TPM'].iloc[:,0:-1]).T],axis=0)
#     dict_DMD_valid['Xf'] = np.concatenate([dict_DMD_valid['Xf'], np.array(dict_MAX[i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
#     dict_DMD_valid['Yp'] = np.concatenate([dict_DMD_valid['Yp'], np.array(dict_MAX[i]['Y'].iloc[:, 0:-1]).T], axis=0)
#     dict_DMD_valid['Yf'] = np.concatenate([dict_DMD_valid['Yf'], np.array(dict_MAX[i]['Y'].iloc[:, 1:]).T], axis=0)

dict_DMD_test = {'Xp' : np.empty(shape=(0,n_states)), 'Xf': np.empty(shape=(0,n_states)),'Yp' : np.empty(shape=(0,n_outputs)), 'Yf' : np.empty(shape=(0,n_outputs))}
for i, COND in itertools.product(ls_test_indices,ALL_CONDITIONS):
    dict_DMD_test['Xp'] = np.concatenate([dict_DMD_test['Xp'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:,0:-1]).T],axis=0)
    dict_DMD_test['Xf'] = np.concatenate([dict_DMD_test['Xf'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
    dict_DMD_test['Yp'] = np.concatenate([dict_DMD_test['Yp'], np.array(dict_data[COND][i]['Y'].iloc[:, 0:-1]).T], axis=0)
    dict_DMD_test['Yf'] = np.concatenate([dict_DMD_test['Yf'], np.array(dict_data[COND][i]['Y'].iloc[:, 1:]).T], axis=0)



SYSTEM_NO = 601
storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
if os.path.exists(storage_folder):
    get_input = input('Do you wanna delete the existing system[y/n]? ')
    # get_input = 'y'
    if get_input == 'y':
        shutil.rmtree(storage_folder)
        os.mkdir(storage_folder)
    else:
        quit(0)
else:
    os.mkdir(storage_folder)

# _, dict_Scaler, _ = oc.scale_train_data(dict_DMD_train, 'standard',WITH_MEAN_FOR_STANDARD_SCALER_X = True, WITH_MEAN_FOR_STANDARD_SCALER_Y = True)
_, dict_Scaler, _ = oc.scale_train_data(dict_DMD_train, 'min max',WITH_MEAN_FOR_STANDARD_SCALER_X = True, WITH_MEAN_FOR_STANDARD_SCALER_Y = True)
# _, dict_Scaler, _ = oc.scale_train_data(dict_DMD_train, 'none',WITH_MEAN_FOR_STANDARD_SCALER_X = True, WITH_MEAN_FOR_STANDARD_SCALER_Y = True)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'wb') as handle:
    pickle.dump(dict_Scaler, handle)
dict_DATA_OUT = oc.scale_data_using_existing_scaler_folder(dict_DMD_train, SYSTEM_NO)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'wb') as handle:
    pickle.dump(dict_DATA_OUT, handle)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_Data.pickle', 'wb') as handle:
    pickle.dump(dict_data, handle)
with open(storage_folder + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle', 'wb') as handle:
    pickle.dump(ls_all_indices, handle)  # Only training and validation indices are stored
# Store the data in Koopman
with open('/Users/shara/Desktop/oc_deepDMD/koopman_data/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle','wb') as handle:
    pickle.dump(dict_DATA_OUT, handle)
