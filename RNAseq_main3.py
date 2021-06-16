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
import re
import copy
import itertools
import seaborn as sb
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

def plot_gene_expression(dict_data):
    # Plotting the states as a function of time
    ls_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    curve = 0
    f,ax = plt.subplots(7,1,sharex=True,figsize=(30,14))
    for time_pt,COND_NO in itertools.product(range(1,8),range(len(ls_conditions))):
        COND = ls_conditions[COND_NO]
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
    return

gene_ontology_file = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/Gene_Ontologies_map.pickle'
with open(gene_ontology_file, 'rb') as handle:
    dict_gene_ontology = pickle.load(handle)
df_GO = dict_gene_ontology['gene ontology']
dict_GOkey = dict_gene_ontology['key for gene ontology']
dict_GOgenes = dict_gene_ontology['gene ontology genes']
# ##
# dict_GOgenes = {}
# for items in dict_GOkey:
#     dict_GOgenes[items] = []
#
# for i in range(df_GO.shape[0]):
#     for items in df_GO.iloc[i,-1]:
#         dict_GOgenes[items].append(df_GO.iloc[i,0])
#
#  =
# with open(gene_ontology_file, 'wb') as handle:
#     pickle.dump(dict_gene_ontology,handle)
##
# keywords =['protein']
keywords = ['carbohydrate','protein','amino acid','glucose', 'cell cycle','division']
# keywords = [keywords[0]]
ls_GO_keywords_filtered = []
for items in dict_GOkey:
    if np.sum([i in dict_GOkey[items] for i in keywords]) > 0:
        ls_GO_keywords_filtered.append(items)

set_filtered_genes = set()
for items in ls_GO_keywords_filtered:
    print(dict_GOkey[items],' # genes :', len(dict_GOgenes[items]))
    print(dict_GOgenes[items])
    set_filtered_genes = set_filtered_genes.union(set(dict_GOgenes[items]))

ls_GO_filtered_genes = list(set_filtered_genes)


## Get the GO filtered data

with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle', 'rb') as handle:
    dict_DATA_ORIGINAL = pickle.load(handle)

ls_conditions = ['MX','MN','NC']
dict_DATA_max_denoised = copy.deepcopy(dict_DATA_ORIGINAL)
dict_data_GO_filtered = {'MX':{},'MN':{},'NC':{}}
for condition,items in itertools.product(ls_conditions,range(16)):
    dict_data_GO_filtered[condition][items] = {'df_X_TPM': dict_DATA_max_denoised[condition][items]['df_X_TPM'].loc[ls_GO_filtered_genes,:], 'Y0': dict_DATA_max_denoised[condition][items]['Y0'], 'Y': dict_DATA_max_denoised[condition][items]['Y']}

plot_gene_expression(dict_data_GO_filtered)
##
dict_data = rnaf.filter_gene_by_coefficient_of_variation(copy.deepcopy(dict_data_GO_filtered), CV_THRESHOLD = 0.02,ALL_CONDITIONS=['MX'])
# dict_data = rnaf.filter_gene_by_coefficient_of_variation(copy.deepcopy(dict_DATA_max_denoised), CV_THRESHOLD = 0.0125,ALL_CONDITIONS=['MX'])
rnaf.formulate_and_save_Koopman_Data(dict_data,SYSTEM_NO= 700, ALL_CONDITIONS= ['MX'])

##

ls_genes_temp2 = list(dict_data['MX'][0]['df_X_TPM'].index)
dict_data_temp = {'MX':{},'MN':{},'NC':{}}

for condition,items in itertools.product(ls_conditions,range(16)):
    dict_data_temp[condition][items] = {'df_X_TPM': dict_DATA_max_denoised[condition][items]['df_X_TPM'].loc[ls_genes_temp2 ,:], 'Y0': dict_DATA_max_denoised[condition][items]['Y0'], 'Y': dict_DATA_max_denoised[condition][items]['Y']}





plot_gene_expression(dict_data_temp )