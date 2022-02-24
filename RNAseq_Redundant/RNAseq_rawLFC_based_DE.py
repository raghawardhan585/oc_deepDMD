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

rnaf.organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True, get_full_output= True,n_outputs= -1) # Getting the raw RNAseq and OD600 data to the state and output data format
with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)


## Getting the raw LFC across time dictionary
n_genes_filterd_across_time = 5
n_genes_filterd_across_cond = 3

DO_LOG_FOLD_CHANGE_ACROSS_TIME = True
DO_LOG_FOLD_CHANGE_ACROSS_CONDITIONS = False

ls_all_cond = ['MX','MN','NC']
ls_all_time = [3,4,5,6,7]
ls_genes_1 = list(dict_DATA_ORIGINAL['MX'][0]['df_X_TPM'].index)
dict_DATA_IN1 = {'MX':{},'MN':{},'NC':{}}
for cond,rep in itertools.product(ls_all_cond, range(0,16)):
    dict_DATA_IN1[cond][rep] = copy.deepcopy(dict_DATA_ORIGINAL[cond][rep]['df_X_TPM'].loc[ls_genes_1,:])+1


if DO_LOG_FOLD_CHANGE_ACROSS_TIME:
    dict_LFC_t = {'MX':{},'MN':{},'NC':{}}
    for cond,rep in itertools.product(ls_all_cond, range(0,16)):
        dict_LFC_t[cond][rep] = copy.deepcopy(dict_DATA_IN1[cond][rep].iloc[:,1:])
        for i in range(dict_LFC_t[cond][rep].shape[1]):
            dict_LFC_t[cond][rep].iloc[:,i] = np.log2(dict_LFC_t[cond][rep].iloc[:,i]/dict_DATA_IN1[cond][rep].iloc[:,0])
    df_LFC_t = {}
    ls_diff_t_locus_tags = []
    for cond in ls_all_cond :
        nx = dict_LFC_t[cond][0].shape[0]
        ny = dict_LFC_t[cond][0].shape[1]
        df_LFC_t[cond] = np.empty(shape=(nx,ny,0))
        for rep in dict_LFC_t[cond].keys():
            df_LFC_t[cond] = np.concatenate([df_LFC_t[cond],np.array(dict_LFC_t[cond][rep]).reshape(nx,ny,1)],axis=2)
        df_LFC_t[cond] = pd.DataFrame(np.median(df_LFC_t[cond],axis=2),index=dict_LFC_t[cond][0].index)
        df_LFC_t[cond]['abs_sum_LFC'] = np.abs(np.sum(df_LFC_t[cond],axis=1))
        df_LFC_t[cond] = df_LFC_t[cond].sort_values(by=['abs_sum_LFC'],ascending=False)
        for items in list((df_LFC_t[cond].index))[0:n_genes_filterd_across_time]:
            ls_diff_t_locus_tags.append(items)
else:
    ls_diff_t_locus_tags = copy.deepcopy(ls_genes_1)

# Sorting the temporally differentiated genes
ls_genes_2 = ls_diff_t_locus_tags
dict_DATA_IN2 = {'MX':{},'MN':{},'NC':{}}
for cond,rep in itertools.product(ls_all_cond, range(0,16)):
    dict_DATA_IN2[cond][rep] = copy.deepcopy(dict_DATA_ORIGINAL[cond][rep]['df_X_TPM'].loc[ls_genes_2,:])+1


if DO_LOG_FOLD_CHANGE_ACROSS_CONDITIONS:
    dict_LFC_x = {3:{},4:{},5:{},6:{},7:{}}
    # for i,j in itertools.combinations(ls_all_cond,2):
    for time,rep in itertools.product(ls_all_time,range(16)):
        dict_LFC_x[time][rep] = pd.DataFrame(np.log2(dict_DATA_IN2['MX'][rep].loc[:,time]/dict_DATA_IN2['NC'][rep].loc[:,time]))
        dict_LFC_x[time][rep] = pd.concat([dict_LFC_x[time][rep],pd.DataFrame(
            np.log2(dict_DATA_IN2['MN'][rep].loc[:, time] / dict_DATA_IN2['NC'][rep].loc[:, time]))],axis=1)
        dict_LFC_x[time][rep].columns = ['MX-NC','MN-NC']

    df_LFC_x = {}
    ls_diff_x_locus_tags = []
    for time in ls_all_time:
        nx = dict_LFC_x[time][0].shape[0]
        ny = dict_LFC_x[time][0].shape[1]
        df_LFC_x[time] = np.empty(shape=(nx,ny,0))
        for rep in dict_LFC_x[time].keys():
            df_LFC_x[time] = np.concatenate([df_LFC_x[time],np.array(dict_LFC_x[time][rep]).reshape(nx,ny,1)],axis=2)
        df_LFC_x[time] = pd.DataFrame(np.median(df_LFC_x[time],axis=2),index=dict_LFC_x[time][0].index,columns =dict_LFC_x[time][0].columns)
        df_LFC_x[time]['abs_sum_LFC'] = np.sum(np.abs(df_LFC_x[time]),axis=1)
        df_LFC_x[time] = df_LFC_x[time].sort_values(by=['abs_sum_LFC'],ascending=False)
        for items in list((df_LFC_x[time].index))[0:n_genes_filterd_across_cond]:
            ls_diff_x_locus_tags.append(items)
        ls_diff_x_locus_tags = list(set(ls_diff_x_locus_tags))
else:
    ls_diff_x_locus_tags = copy.deepcopy(ls_genes_2)


print(ls_diff_x_locus_tags)
# ls_diff_x_locus_tags.append('PP_1733')
## Gene ontologies of the various genes
df_gene_name = rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags = ls_diff_x_locus_tags,search_columns='genes(OLN),genes(PREFERRED)')

print('-----------------------')
for gene_locus_tag in ls_diff_x_locus_tags:
    print('Gene Locus Tag :',gene_locus_tag,' | Gene Name :',df_gene_name[df_gene_name['Gene names  (ordered locus )']== gene_locus_tag].iloc[-1,-1])

df_gene_name.columns =['locus tag','gene']
df_gene_name.index = df_gene_name['locus tag']
for i in range(df_gene_name.shape[0]):
    if df_gene_name.iloc[i,1] == '':
        df_gene_name.iloc[i, 1] =  df_gene_name.iloc[i, 0]



##

ny = np.int(np.ceil(np.sqrt(len(ls_diff_x_locus_tags))))
nx = np.int(np.ceil(len(ls_diff_x_locus_tags)/ny))
ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

f,ax = plt.subplots(nrows = nx,ncols=ny,sharex= True,figsize = (15,10))
ls_ax = ax.reshape(-1)
for i in range(len(ls_diff_x_locus_tags)):
    gene = ls_diff_x_locus_tags[i]
    for cond in ls_all_cond:
        time = list(dict_DATA_ORIGINAL[cond][0]['df_X_TPM'].columns)
        data_inst = np.array(dict_DATA_ORIGINAL[cond][0]['df_X_TPM'].loc[gene,:]).reshape(1,-1)
        for rep in range(1,16):
            data_inst = np.concatenate([data_inst,np.array(dict_DATA_ORIGINAL[cond][rep]['df_X_TPM'].loc[gene,:]).reshape(1,-1)],axis=0)
        if cond == 'MX':
            color = ls_colors[0]
            label = 'MAX growth'
        elif cond == 'MN':
            color = ls_colors[1]
            label = 'MIN growth'
        else:
            color = ls_colors[2]
            label = 'NC growth'
        mean = np.mean(data_inst, axis=0)
        standard_dev = np.std(data_inst, axis=0)
        ls_ax[i].plot(time, mean, color=color, label =label)
        ls_ax[i].fill_between(time, mean - standard_dev, mean + standard_dev, alpha=0.2)
    print(gene)
    ls_ax[i].set_title(df_gene_name.loc[gene, 'gene'] + '(' + df_gene_name.loc[gene, 'locus tag'] + ')')
    ls_ax[i].set_xlabel('time (hrs)')
    ls_ax[i].set_ylabel('Expression (TPM)')
    if i==0:
        ls_ax[i].legend()

f.show()


# for gene in ls_diff_x_locus_tags:
#     print('gene =',gene)
#     print(dict_DATA_IN1['MX'][0].loc[gene,:])
#     print(dict_DATA_IN1['MN'][0].loc[gene,:])
#     print(dict_DATA_IN1['NC'][0].loc[gene,:])


##

df_gene_func = rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags = ls_diff_x_locus_tags,search_columns='genes(OLN),genes(PREFERRED),go(biological process),go(molecular function)')