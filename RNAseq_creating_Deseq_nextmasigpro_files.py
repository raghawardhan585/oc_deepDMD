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
import re

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22

MAX_REPLICATES = 2
MAX_LANES = 4
MAX_READS = 2
RNA_FILE_LOCATION = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/raw_rnaseq_files'


# Processing the RNAseq data
dict_DATA = {'MX':{},'MN':{},'NC':{}}
ls_RNA_files = os.listdir(RNA_FILE_LOCATION)
for file in ls_RNA_files:
    header = re.split('\s|-|_|/', file)
    cond_name = header[0][0:2]
    read_no = np.int(header[4][1])
    lane_no = np.int(header[3][3])
    rep_no = np.int(header[0][-1])
    time_pt = np.int(header[1])
    df_RAW = pd.read_csv(RNA_FILE_LOCATION + '/' + file)
    curve_no = rnaf.get_curve_dict_number(rep_no,lane_no,read_no)
    # Processing the current dataframe
    df_temp = copy.deepcopy(df_RAW)
    df_temp.index = df_temp.locus_tag
    # df_TPM = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'TPM']))
    # df_TPM = df_TPM.rename(columns={'TPM': time_pt})
    df_raw_count = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'Raw Fragment Count']))
    df_raw_count = df_raw_count.rename(columns={'Raw Fragment Count': time_pt})
    # df_FPKM = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'FPKM']))
    # df_FPKM = df_FPKM.rename(columns={'FPKM': time_pt})
    # df_RPKM = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'RPKM']))
    # df_RPKM = df_RPKM.rename(columns={'RPKM': time_pt})
    if curve_no in dict_DATA[cond_name].keys():
        # Add data to existing dataframe
        dict_DATA[cond_name][curve_no]['raw_counts'] = pd.concat([dict_DATA[cond_name][curve_no]['raw_counts'],df_raw_count],axis = 1)
        # dict_DATA[cond_name][curve_no]['df_X_FPKM'] = pd.concat([dict_DATA[cond_name][curve_no]['df_X_FPKM'], df_FPKM], axis=1)
        # dict_DATA[cond_name][curve_no]['df_X_RPKM'] = pd.concat([dict_DATA[cond_name][curve_no]['df_X_RPKM'], df_RPKM], axis=1)
    else:
        # Create data with new dataframe
        dict_DATA[cond_name][curve_no]={}
        dict_DATA[cond_name][curve_no]['raw_counts'] = df_raw_count
        # dict_DATA[cond_name][curve_no]['df_X_TPM'] = df_TPM
        # dict_DATA[cond_name][curve_no]['df_X_FPKM'] = df_FPKM
        # dict_DATA[cond_name][curve_no]['df_X_RPKM'] = df_RPKM

for cond in dict_DATA.keys():
    for curve_no in dict_DATA[cond].keys():
        # Sort the dataframes
        dict_DATA[cond][curve_no]['raw_counts'] = np.ceil(dict_DATA[cond][curve_no]['raw_counts'].reindex(
            sorted(dict_DATA[cond][curve_no]['raw_counts'].columns), axis=1))

## Creating a metadata file
FIELD = 'raw_counts'
ngenes = len(dict_DATA['MX'][0][FIELD].index)
dict_metadata = {}
np_data = np.empty(shape=(ngenes,0))
sample_no = 0
for cond in ['MX','MN','NC']:
    for t in dict_DATA[cond][rep][FIELD].columns:
        sample_no = sample_no + 1
        for rep in range(16):
            field = cond + '_R' + str(rep) + '_' + str(t) + 'H'
            dict_metadata[field] = {}
            # Field 1 - Time
            dict_metadata[field]['Time'] = t
            # Field 2 - Replicate
            dict_metadata[field]['Replicate'] = sample_no
            # Field 3 - Control
            if cond == 'NC':
                dict_metadata[field]['Control'] = 1
            else:
                dict_metadata[field]['Control'] = 0
            # Field 4 - Max Growth
            if cond == 'MX':
                dict_metadata[field]['MAX'] = 1
            else:
                dict_metadata[field]['MAX'] = 0
            # Field 5 - Min Growth
            if cond == 'MN':
                dict_metadata[field]['MIN'] = 1
            else:
                dict_metadata[field]['MIN'] = 0
            np_data = np.concatenate([np_data, np.array(dict_DATA[cond][rep][FIELD].loc[:,t]).reshape(-1,1)], axis=1)
pd.DataFrame(dict_metadata).T.to_csv('/Users/shara/Desktop/masigpro_RNAseq/metadata_nextmasigpro_Shara_X1.csv', index_label= None)
pd.DataFrame(np_data,index=list(dict_DATA['MX'][0][FIELD].index),columns=list(dict_metadata.keys())).to_csv('/Users/shara/Desktop/masigpro_RNAseq/data_nextmasigpro_Shara_X1.csv')
