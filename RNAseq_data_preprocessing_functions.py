import pandas as pd
import numpy as np
import re # For splitting the strings
from glob import glob as GetAllCsvFiles # To access
import os
import itertools
# from bioservices import UniProt
import copy
import pickle

# Constants
MAX_REPLICATES = 2
MAX_LANES = 4
MAX_READS = 2
RNA_FILE_LOCATION = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/raw_rnaseq_files'
OD_FILE = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/raw_od_file/PPutida_R2A_Growth2_MIN_MAX_NC_RNASeq.txt'

# SOME KEYWORD USAGES
# np - either an np array or nd array
# ls - list object
# dict - dictionary object
# df - pandas dataframe object
# pseudo - intermediate variable

def get_curve_dict_number(replicate_number,lane_number,read_no):
    return (replicate_number-1)*MAX_LANES*MAX_READS + (lane_number-1)*MAX_READS + read_no - 1

def get_curve_info_from_dict_number(dict_no):
    read_no = np.mod(dict_no,MAX_READS) + 1
    lane_no = np.floor(dict_no/MAX_READS)
    lane_no = np.mod(lane_no,MAX_LANES) + 1
    rep_no = np.floor(dict_no/MAX_READS/MAX_LANES)
    rep_no = np.mod(rep_no,MAX_REPLICATES) + 1
    return np.int(rep_no),np.int(lane_no),np.int(read_no)

def Pputida_R2A_RNAseq_metadata():
    # Input [All Units in g/L]
    dict_input = {'MX': {}, 'MN': {}, 'NC': {}}
    dict_input['MX']['Casein'] = 3.51  # g/L
    dict_input['MX']['Glucose'] = 0.146  # g/L
    dict_input['MN']['Casein'] = 112.5  # g/L
    dict_input['MN']['Glucose'] = 150  # g/L
    dict_input['NC']['Casein'] = 0  # g/L
    dict_input['NC']['Glucose'] = 0  # g/L

    MAX1_WELL = ['A3', 'B3', 'C3', 'D3', 'E3', 'F3', 'G3', 'H3']
    MAX2_WELL = ['A4', 'B4', 'C4', 'D4', 'E4', 'F4', 'G4', 'H4']
    MIN1_WELL = ['A6', 'B6', 'C6', 'D6', 'E6', 'F6', 'G6', 'H6']
    MIN2_WELL = ['A7', 'B7', 'C7', 'D7', 'E7', 'F7', 'G7', 'H7']
    NC1_WELL = ['A9', 'B9', 'C9', 'D9', 'E9', 'F9', 'G9', 'H9']
    NC2_WELL = ['A10', 'B10', 'C10', 'D10', 'E10', 'F10', 'G10', 'H10']
    dict_well = {'MX': {1: MAX1_WELL, 2: MAX2_WELL}, 'MN': {1: MIN1_WELL, 2: MIN2_WELL},
                 'NC': {1: NC1_WELL, 2: NC2_WELL}}
    return dict_input,dict_well

def process_microplate_reader_txtfile(filename):
    RawData = []
    with open(filename, "r") as f:  # 'r' is used to specify Read Mode
        reader = f.readlines()
        linecount = len(reader)
        # Skip everything upto the point where our useful data is available
        tim_ct = 0
        for i in range(linecount):
            iteritem = re.split(r'[\t\s]\s*', reader[i])  # re package is used just for this purpose
            if (iteritem[0] == "Time"):#if (iteritem[0] == "Time"):
                tim_ct = tim_ct + 1
                if tim_ct == 2:
                    istop = i
                    break
        iteritem.remove(iteritem[2])
        # iteritem.remove(iteritem[2])
        iteritem.remove(iteritem[-1])
        # Creating the pandas Dataframe
        RawData.append(iteritem)
        for i in range(istop + 1, linecount):
            iteritem = re.split(r'[\t\s]\s*', reader[i])  # re package is used just for this purpose
            iteritem.remove(iteritem[-1])
            if (iteritem[0] in ["","Results", "0:00:00"]):
                break;
            RawData.append(iteritem)
        Tb = pd.DataFrame(RawData[1:], columns=RawData[0])
    f.close()
    Tb = Tb.drop(columns='T∞',axis =1) # Dropping the T∞ column
    T1 = Tb.iloc[0,0]
    T2 = Tb.iloc[1,0]
    t2_str = re.split(':', T2)
    t2 = int(t2_str[0]) * 3600 + int(t2_str[1]) * 60 + int(t2_str[2])
    t1_str = re.split(':', T1)
    t1 = int(t1_str[0]) * 3600 + int(t1_str[1]) * 60 + int(t1_str[2])
    Ts_sec = t2 - t1
    Time = Tb.loc[:,'Time']
    Tb = Tb.drop(columns='Time',axis =1) # Dropping the T∞ column
    df_OD_DATA = Tb
    df_OD_DATA.index = df_OD_DATA.index*Ts_sec/3600
    # NOTE: The indices of df_DATA indicate the time of measurement in hours
    return df_OD_DATA




def organize_RNAseq_OD_to_RAWDATA():
    # Processing the OD data - Inupts are IGNORED
    # TODO - Include the inputs of Casein and Glucose if need be for later
    df_OD_RAW = process_microplate_reader_txtfile(OD_FILE)
    dict_OD = {'MX':{},'MN':{},'NC':{}}
    _ , dict_well = Pputida_R2A_RNAseq_metadata()
    for COND in dict_well.keys():
        i = 0
        for REP in range(1,3):
            for well in dict_well[COND][REP]:
                df_temp = copy.deepcopy(pd.DataFrame(df_OD_RAW.loc[:,well]))
                # Formulate  and assign the new matrix
                n_outputs = np.sum(np.array(df_temp.index)<1) # We assume that the measurement starts at t=0
                y_temp = np.asarray(df_temp.iloc[:, :], dtype='f').reshape(-1)
                N_samples = np.int(np.floor(len(y_temp)/n_outputs))
                dict_OD[COND][i] = pd.DataFrame(y_temp[0:N_samples*n_outputs].reshape((-1,n_outputs)).T)
                i = i+1

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
        curve_no = get_curve_dict_number(rep_no,lane_no,read_no)
        # Processing the current dataframe
        df_temp = copy.deepcopy(df_RAW)
        df_temp.index = df_temp.locus_tag
        df_TPM = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'TPM']))
        df_TPM = df_TPM.rename(columns={'TPM': time_pt})
        # df_FPKM = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'FPKM']))
        # df_FPKM = df_FPKM.rename(columns={'FPKM': time_pt})
        # df_RPKM = copy.deepcopy(pd.DataFrame(df_temp.loc[:, 'RPKM']))
        # df_RPKM = df_RPKM.rename(columns={'RPKM': time_pt})
        if curve_no in dict_DATA[cond_name].keys():
            # Add data to existing dataframe
            dict_DATA[cond_name][curve_no]['df_X_TPM'] = pd.concat([dict_DATA[cond_name][curve_no]['df_X_TPM'],df_TPM],axis = 1)
            # dict_DATA[cond_name][curve_no]['df_X_FPKM'] = pd.concat([dict_DATA[cond_name][curve_no]['df_X_FPKM'], df_FPKM], axis=1)
            # dict_DATA[cond_name][curve_no]['df_X_RPKM'] = pd.concat([dict_DATA[cond_name][curve_no]['df_X_RPKM'], df_RPKM], axis=1)
        else:
            # Create data with new dataframe
            dict_DATA[cond_name][curve_no]={}
            dict_DATA[cond_name][curve_no]['df_X_TPM'] = df_TPM
            # dict_DATA[cond_name][curve_no]['df_X_FPKM'] = df_FPKM
            # dict_DATA[cond_name][curve_no]['df_X_RPKM'] = df_RPKM

    # Sort the dataframes and join the state and output data
    for cond in dict_DATA.keys():
        for curve_no in dict_DATA[cond].keys():
            # Sort the dataframes
            dict_DATA[cond][curve_no]['df_X_TPM'] = dict_DATA[cond][curve_no]['df_X_TPM'].reindex(sorted( dict_DATA[cond][curve_no]['df_X_TPM'].columns),axis=1)
            # Assign the output
            dict_DATA[cond][curve_no]['Y'] = copy.deepcopy(dict_OD[cond][curve_no].loc[:,dict_DATA[cond][curve_no]['df_X_TPM'].columns])
    # Saving the data
    with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle','wb') as handle:
        pickle.dump(dict_DATA,handle)
    return


# ====================================================================================================================
# Gene Filtering Functions
# ====================================================================================================================
def filter_gene_by_coefficient_of_variation(dict_GrowthCurve, CV_THRESHOLD = np.Inf, MEAN_TPM_THRESHOLD = -np.Inf, ALL_CONDITIONS = ['MX','MN','NC']):
    # Get the gene locustag list
    temp_cond = list(dict_GrowthCurve.keys())[0]
    temp_curve = list(dict_GrowthCurve[temp_cond].keys())[0]
    ls_GENE_NAME = list(dict_GrowthCurve[temp_cond][temp_curve]['df_X_TPM'].index)
    print('---------------------------------------------------')
    print('FILTER - COEFFICIENT OF VARIATION [INVERSE OF SNR]')
    # Get the coefficient of variation
    temp_np = np.empty(shape=(len(ls_GENE_NAME), 0))
    for COND in ALL_CONDITIONS:
        for CURVE in dict_GrowthCurve[COND]:
            temp_np = np.concatenate([temp_np,np.array(dict_GrowthCurve[COND][CURVE]['df_X_TPM'])],axis=1)
    temp_CV = temp_np.std(axis=1)/temp_np.mean(axis=1)
    ls_GENE_ALLOW = [ls_GENE_NAME[i] for i in range(len(ls_GENE_NAME)) if temp_CV[i]<CV_THRESHOLD]
    ls_GENE_REMOVE = list(set(ls_GENE_NAME) - set(ls_GENE_ALLOW))
    print('The number of removed genes:',len(ls_GENE_REMOVE))
    print('Remaining Genes:',len(ls_GENE_ALLOW))
    print('---------------------------------------------------')
    print('FILTER - MEAN ')
    temp_mean = temp_np.mean(axis=1)
    ls_GENE_ALLOW2 = [ls_GENE_ALLOW[i] for i in range(len(ls_GENE_ALLOW)) if temp_mean[i] > MEAN_TPM_THRESHOLD]
    ls_GENE_REMOVE2 = list(set(ls_GENE_ALLOW) - set(ls_GENE_ALLOW2))
    print('The number of removed genes:', len(ls_GENE_REMOVE2))
    print('Remaining Genes:', len(ls_GENE_ALLOW2))
    print('---------------------------------------------------')
    for COND in ALL_CONDITIONS:
        for CURVE in dict_GrowthCurve[COND]:
            dict_GrowthCurve[COND][CURVE]['df_X_TPM'] = dict_GrowthCurve[COND][CURVE]['df_X_TPM'].drop(ls_GENE_REMOVE,axis=0)
            dict_GrowthCurve[COND][CURVE]['df_X_TPM'] = dict_GrowthCurve[COND][CURVE]['df_X_TPM'].drop(ls_GENE_REMOVE2,axis=0)
    return dict_GrowthCurve

def filter_gene_by_threshold(dict_GrowthCurve,MEAN_TPM_THRESHOLD = 10, ALL_CONDITIONS = ['MAX','MIN','NC']):
    # Downselect the genes that have zero gene expression across all datasets
    df_GENE_SCORE = pd.DataFrame([], columns=dict_GrowthCurve['MAX'][3].RNAdata.columns)
    for COND, REPLICATE in itertools.product(ALL_CONDITIONS, range(1, 17, 1)):
        df_GENE_SCORE = df_GENE_SCORE.append(dict_GrowthCurve[COND][REPLICATE].RNAdata)
    df_GENE_SCORE = df_GENE_SCORE.mean(axis=0)
    ls_GENE_REJECT = []
    for gene in df_GENE_SCORE.index:
        if (df_GENE_SCORE[gene] < MEAN_TPM_THRESHOLD):
            ls_GENE_REJECT.append(gene)
    print('The number of removed genes:', len(ls_GENE_REJECT))
    for COND, REPLICATE in itertools.product(ALL_CONDITIONS, range(1, 17, 1)):
        dict_GrowthCurve[COND][REPLICATE].RNAdata = dict_GrowthCurve[COND][REPLICATE].RNAdata.drop(ls_GENE_REJECT,axis=1)
    print('Remaining Genes:', dict_GrowthCurve[COND][REPLICATE].RNAdata.shape[1])
    return dict_GrowthCurve





# from bioservices import UniProt
# # ====================================================================================================================
# # Functions to query Uniprot
# # ====================================================================================================================
# # The file format is ALWAYS set to tab delimited
# # For the valid column names refer to the website: https://www.uniprot.org/help/uniprotkb_column_names
# def get_gene_Uniprot_DATA(species_id='KT2440', ls_all_locus_tags='PP_0123',
#                           search_columns='entry name,length,id, genes,comment(FUNCTION)'):
#     query_search =''
#     for locus_tag in ls_all_locus_tags[0:-1]:
#         query_search = query_search + locus_tag + ' OR '
#     query_search = query_search + ls_all_locus_tags[-1] + ' AND ' + species_id
#     up = UniProt()
#     search_result = up.search(query_search, frmt='tab',columns=search_columns)

#     # Creating the dataframe with the obtained entries
#     # up - uniprot
#     str_up_ALL = search_result.split('\n')
#     ls_up = []
#     for each_line in str_up_ALL[1:]:
#         ls_up.append(each_line.split('\t'))
#     df_up = pd.DataFrame(ls_up[0:-1])
#     df_up.columns = str_up_ALL[0].split('\t')
#     return df_up







