##
import RNAseq_data_preprocessing_functions as rnaf
import ocdeepdmd_simulation_examples_helper_functions as oc
import re
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

# ALL_CONDITIONS = ['MX','NC','MN']
#
# dict_data = rnaf.filter_gene_by_coefficient_of_variation(dict_DATA_max_denoised, CV_THRESHOLD=1, ALL_CONDITIONS=ALL_CONDITIONS)
ls_genes = list(dict_DATA_max_denoised['MX'][0]['df_X_TPM'].index)
##
n_genes_per_run = 400
df_gene_ont = pd.DataFrame([])
for i in range(0,len(ls_genes),n_genes_per_run):
    print('Gene :',i)
    df_inter = rnaf.get_gene_Uniprot_DATA(ls_all_locus_tags=ls_genes[i:i+n_genes_per_run], search_columns='genes(OLN), genes(PREFERRED), go(biological process)')
    try:
        df_gene_ont = pd.concat([df_gene_ont, copy.deepcopy(df_inter)],axis=0)
    except:
        df_gene_ont = copy.deepcopy(df_inter)

##
df_gene_ont1 = copy.deepcopy(df_gene_ont)
dict_temp = {}
for i in range(df_gene_ont1.shape[0]):
    # Make all the gene ontologies into a list
    ls_bio_ont_temp = re.split(';', df_gene_ont1.iloc[i, -1])
    # concatenate the gene ontologies
    ls_gene_ont_temp =[]
    for items in ls_bio_ont_temp:
        if not(items == ''):
            ls_temp = re.split('\[|\]', items)
            dict_temp[ls_temp[1]] = ls_temp[0]
            ls_gene_ont_temp.append(ls_temp[1])
    df_gene_ont1.iloc[i,-1] = ls_gene_ont_temp
    df_gene_ont1.iloc[i, 0] = re.split(' ',df_gene_ont1.iloc[i, 0])

# eliminiating rows with multiple locus tags
df_gene_ont2 = pd.DataFrame([],columns=df_gene_ont1.columns)
for i in range(df_gene_ont1.shape[0]):
    for items in df_gene_ont1.iloc[i,0]:
        df_gene_ont2 = pd.concat([df_gene_ont2,df_gene_ont1.iloc[i:i+1, :]],axis=0)
        df_gene_ont2.iloc[-1,0] = items

# eliminiating duplicity of locus tags
df_gene_ont3 = pd.DataFrame([],columns=['locus tag','gene ontology (biological process)'])
for locus_tag in ls_genes:
    df_temp = df_gene_ont2[df_gene_ont2.iloc[:,0] == locus_tag]
    ls_gene_ont_temp = []
    for i in range(df_temp.shape[0]):
        for items in df_temp.iloc[i,-1]:
            ls_gene_ont_temp.append(items)
    df_gene_ont3 = df_gene_ont3.append({'locus tag':locus_tag,'gene ontology (biological process)':ls_gene_ont_temp}, ignore_index=True)

# Saving the genes corresponding to each
dict_GOgenes = {}
for items in dict_temp:
    dict_GOgenes[items] = []
for i in range(df_gene_ont3.shape[0]):
    for items in df_gene_ont3.iloc[i,-1]:
        dict_GOgenes[items].append(df_gene_ont3.iloc[i,0])

dict_out = {}
dict_out['gene ontology'] = df_gene_ont3
dict_out['key for gene ontology'] = dict_temp
dict_out['gene ontology genes'] = dict_GOgenes

with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/Gene_Ontologies_map.pickle', 'wb') as handle:
    pickle.dump(dict_out, handle)



