import pandas as pd
import numpy as np
import re # For splitting the strings
from glob import glob as GetAllCsvFiles # To access
import os
import shutil
import itertools
# from bioservices import UniProt
import copy
import pickle
import matplotlib.pyplot as plt
import random
import ocdeepdmd_simulation_examples_helper_functions as oc
import tensorflow as tf
from sklearn.metrics import r2_score
import seaborn as sb

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

def get_PputidaKT2440_growth_genes():
    # Pseudomonas putida Biocyc genes specific
    cell_division_filename = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/cell_division_genes.txt'
    cell_cycle_filename = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/cell_cycle_genes.txt'
    dict_biocyc_growth_genes = {}
    for filename in [cell_cycle_filename,cell_division_filename]:
        f = open(filename, "r")
        reader = f.readlines()
        ls_genes =[]
        for line in reader[0:-1]:
            last_word = re.split(r'\s',line)[-2].replace('(','').replace(')', '').replace(',', '')
            if not(last_word[0:3] == 'PP_'):
                last_word = last_word[0].lower() + last_word[1:]
            ls_genes.append(last_word)
        last_word = re.split(r'\s',reader[-1])[-1].replace('(', '').replace(')', '').replace(',', '')
        ls_genes.append(last_word)
        f.close()
        # mapping the gene names to the locus tags
        df_RAW = pd.read_csv(RNA_FILE_LOCATION + '/' + os.listdir(RNA_FILE_LOCATION)[0])
        dict_locus_tags = {}
        for i in range(df_RAW.shape[0]):
            if not(df_RAW.loc[i,'gene'] =='NaN'):
                dict_locus_tags[df_RAW.loc[i,'gene']] = df_RAW.loc[i,'locus_tag']
        # getting all the gene locus tags
        ls_locus_tags = []
        for i in range(len(ls_genes)):
            if ls_genes[i][0:3] == 'PP_':
                ls_locus_tags.append(ls_genes[i])
            elif ls_genes[i] in dict_locus_tags.keys():
                ls_locus_tags.append(dict_locus_tags[ls_genes[i]])
            else:
                print('[ERROR]: The gene ', i , ' labeled ', ls_genes[i],' could not be located and is not added')
        if  filename ==cell_cycle_filename:
            dict_biocyc_growth_genes['cell_cycle'] = ls_locus_tags
        elif filename == cell_division_filename:
            dict_biocyc_growth_genes['cell_division'] = ls_locus_tags
    return dict_biocyc_growth_genes




def get_curve_dict_number(replicate_number,lane_number,read_no):
    return (replicate_number-1)*MAX_LANES*MAX_READS + (lane_number-1)*MAX_READS + read_no - 1

def get_curve_info_from_dict_number(dict_no):
    read_no = np.mod(dict_no,MAX_READS) + 1
    lane_no = np.floor(dict_no/MAX_READS)
    lane_no = np.mod(lane_no,MAX_LANES) + 1
    rep_no = np.floor(dict_no/MAX_READS/MAX_LANES)
    rep_no = np.mod(rep_no,MAX_REPLICATES) + 1
    print('Replicate No:', np.int(rep_no))
    print('Lane No:', np.int(lane_no))
    print('Read No:', np.int(read_no))
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

def formulate_and_save_Koopman_Data(dict_data,ALL_CONDITIONS =['MX'],SYSTEM_NO=0):
    ls_all_indices = list(dict_data[ALL_CONDITIONS[0]].keys())
    random.shuffle(ls_all_indices)
    ls_train_indices = ls_all_indices[0:14]
    # ls_valid_indices = ls_all_indices[12:14]
    ls_test_indices = ls_all_indices[14:16]
    n_states = dict_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['df_X_TPM'].shape[0]
    n_outputs = dict_data[ALL_CONDITIONS[0]][ls_train_indices[0]]['Y'].shape[0]

    dict_DMD_train = {'Xp': np.empty(shape=(0, n_states)), 'Xf': np.empty(shape=(0, n_states)),
                      'Yp': np.empty(shape=(0, n_outputs)), 'Yf': np.empty(shape=(0, n_outputs))}
    for i, COND in itertools.product(ls_train_indices, ALL_CONDITIONS):
        dict_DMD_train['Xp'] = np.concatenate([dict_DMD_train['Xp'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T], axis=0)
        dict_DMD_train['Xf'] = np.concatenate([dict_DMD_train['Xf'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
        dict_DMD_train['Yp'] = np.concatenate([dict_DMD_train['Yp'], np.array(dict_data[COND][i]['Y'].iloc[:, 0:-1]).T], axis=0)
        dict_DMD_train['Yf'] = np.concatenate([dict_DMD_train['Yf'], np.array(dict_data[COND][i]['Y'].iloc[:, 1:]).T], axis=0)

    # dict_DMD_test = {'Xp': np.empty(shape=(0, n_states)), 'Xf': np.empty(shape=(0, n_states)),
    #                  'Yp': np.empty(shape=(0, n_outputs)), 'Yf': np.empty(shape=(0, n_outputs))}
    # for i, COND in itertools.product(ls_test_indices, ALL_CONDITIONS):
    #     dict_DMD_test['Xp'] = np.concatenate([dict_DMD_test['Xp'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T], axis=0)
    #     dict_DMD_test['Xf'] = np.concatenate([dict_DMD_test['Xf'], np.array(dict_data[COND][i]['df_X_TPM'].iloc[:, 1:]).T], axis=0)
    #     dict_DMD_test['Yp'] = np.concatenate([dict_DMD_test['Yp'], np.array(dict_data[COND][i]['Y'].iloc[:, 0:-1]).T],axis=0)
    #     dict_DMD_test['Yf'] = np.concatenate([dict_DMD_test['Yf'], np.array(dict_data[COND][i]['Y'].iloc[:, 1:]).T],axis=0)

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
    _, dict_Scaler, _ = oc.scale_train_data(dict_DMD_train, 'min max', WITH_MEAN_FOR_STANDARD_SCALER_X=True, WITH_MEAN_FOR_STANDARD_SCALER_Y=True)
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
    return

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


def get_dataframe_with_differenced_data(df_IN):
    df_indices = list(df_IN.index)
    df_columns = list(df_IN.columns)
    np_diff = np.concatenate([np.array([0]), np.diff(df_IN.to_numpy().T.reshape(-1))], axis=0)
    df_OUT = pd.DataFrame(np_diff.reshape(len(df_columns),len(df_indices)).T,columns = df_columns,index = df_indices)
    return df_OUT

def organize_RNAseq_OD_to_RAWDATA(get_fitness_output = True,n_outputs =-1):
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
                n_outputs_per_hour = np.sum(np.array(df_temp.index)<1) # We assume that the measurement starts at t=0
                y_temp = np.asarray(df_temp.iloc[:, :], dtype='f').reshape(-1)
                N_samples = np.int(np.floor(len(y_temp)/n_outputs_per_hour))
                if not((type(n_outputs) == int) and (n_outputs>0)):
                    n_outputs = n_outputs_per_hour
                dict_OD[COND][i] = pd.DataFrame(y_temp[0:N_samples*n_outputs_per_hour].reshape((-1,n_outputs_per_hour)).T).iloc[0:n_outputs,:]
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
            dict_DATA[cond][curve_no]['Y0'] = dict_OD[cond][curve_no].iloc[0,0]
            # dict_DATA[cond][curve_no]['Y0'] = dict_OD[cond][curve_no].loc[:,np.int(np.min(dict_DATA[cond][curve_no]['df_X_TPM'].columns))-1].to_numpy()[-1]
            if get_fitness_output:
                dict_DATA[cond][curve_no]['Y'] = np.log2(dict_OD[cond][curve_no].loc[:, dict_DATA[cond][curve_no]['df_X_TPM'].columns]/dict_DATA[cond][curve_no]['Y0'])
                # df_intermediate = get_dataframe_with_differenced_data(dict_OD[cond][curve_no])
                # dict_DATA[cond][curve_no]['Y'] = copy.deepcopy(df_intermediate.loc[:,dict_DATA[cond][curve_no]['df_X_TPM'].columns])
            else:
                dict_DATA[cond][curve_no]['Y'] = copy.deepcopy(dict_OD[cond][curve_no].loc[:, dict_DATA[cond][curve_no]['df_X_TPM'].columns])
    # Saving the data
    with open('/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/dict_XYData_RAW.pickle','wb') as handle:
        pickle.dump(dict_DATA,handle)
    return


# ====================================================================================================================
# Functions to get scaled and unscaled training,test and validation data
# ====================================================================================================================
def get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS = ['MX']):
    # The split is kept at 14 curves for training, 2 curves for validation and  2 curves for testing
    original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
        SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
    indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
        SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
    root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
        SYSTEM_NO)

    # Scaler import
    with open(root_run_file + '/System_' + str(SYSTEM_NO) + '_DataScaler.pickle', 'rb') as handle:
        All_Scalers = pickle.load(handle)
    X_scaler = All_Scalers['X Scale']
    Y_scaler = All_Scalers['Y Scale']

    # Indices [train, validation and test]
    with open(indices_path, 'rb') as handle:
        ls_data_indices = pickle.load(handle)
    ls_train_indices = ls_data_indices[0:12]
    ls_valid_indices = ls_data_indices[12:14]
    ls_test_indices = ls_data_indices[14:16]
    # Datasets [sorted as scaled and unscaled] and Conditions
    with open(original_data_path, 'rb') as handle:
        dict_data_original = pickle.load(handle)

    dict_empty_all_conditions = {}
    for COND in ALL_CONDITIONS:
        dict_empty_all_conditions[COND] = {}

    dict_scaled_data = copy.deepcopy(dict_empty_all_conditions)
    dict_unscaled_data = copy.deepcopy(dict_empty_all_conditions)
    for i,COND in itertools.product(ls_data_indices, ALL_CONDITIONS):
        dict_unscaled_data[COND][i] = {'XpT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 0:-1]).T,
                                       'XfT': np.array(dict_data_original[COND][i]['df_X_TPM'].iloc[:, 1:]).T,
                                       'YpT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 0:-1]).T,
                                       'YfT': np.array(dict_data_original[COND][i]['Y'].iloc[:, 1:]).T}
        dict_scaled_data[COND][i] = {'XpT': X_scaler.transform(dict_unscaled_data[COND][i]['XpT']),
                                     'XfT': X_scaler.transform(dict_unscaled_data[COND][i]['XfT']),
                                     'YpT': Y_scaler.transform(dict_unscaled_data[COND][i]['YpT']),
                                     'YfT': Y_scaler.transform(dict_unscaled_data[COND][i]['YfT'])}

    XpTs_train = XfTs_train = XpTs_valid = XfTs_valid = XpTs_test = XfTs_test = []
    YpTs_train = YfTs_train = YpTs_valid = YfTs_valid = YpTs_test = YfTs_test = []
    for i,COND in itertools.product(ls_train_indices, ALL_CONDITIONS):
        try:
            XpTs_train = np.concatenate([XpTs_train, dict_scaled_data[COND][i]['XpT']], axis=0)
            XfTs_train = np.concatenate([XfTs_train, dict_scaled_data[COND][i]['XfT']], axis=0)
            YpTs_train = np.concatenate([YpTs_train, dict_scaled_data[COND][i]['YpT']], axis=0)
            YfTs_train = np.concatenate([YfTs_train, dict_scaled_data[COND][i]['YfT']], axis=0)
        except:
            XpTs_train = dict_scaled_data[COND][i]['XpT']
            XfTs_train = dict_scaled_data[COND][i]['XfT']
            YpTs_train = dict_scaled_data[COND][i]['YpT']
            YfTs_train = dict_scaled_data[COND][i]['YfT']

    for i,COND in itertools.product(ls_valid_indices, ALL_CONDITIONS):
        try:
            XpTs_valid = np.concatenate([XpTs_valid, dict_scaled_data[COND][i]['XpT']], axis=0)
            XfTs_valid = np.concatenate([XfTs_valid, dict_scaled_data[COND][i]['XfT']], axis=0)
            YpTs_valid = np.concatenate([YpTs_valid, dict_scaled_data[COND][i]['YpT']], axis=0)
            YfTs_valid = np.concatenate([YfTs_valid, dict_scaled_data[COND][i]['YfT']], axis=0)
        except:
            XpTs_valid = dict_scaled_data[COND][i]['XpT']
            XfTs_valid = dict_scaled_data[COND][i]['XfT']
            YpTs_valid = dict_scaled_data[COND][i]['YpT']
            YfTs_valid = dict_scaled_data[COND][i]['YfT']

    for i,COND in itertools.product(ls_test_indices, ALL_CONDITIONS):
        try:
            XpTs_test = np.concatenate([XpTs_test, dict_scaled_data[COND][i]['XpT']], axis=0)
            XfTs_test = np.concatenate([XfTs_test, dict_scaled_data[COND][i]['XfT']], axis=0)
            YpTs_test = np.concatenate([YpTs_test, dict_scaled_data[COND][i]['YpT']], axis=0)
            YfTs_test = np.concatenate([YfTs_test, dict_scaled_data[COND][i]['YfT']], axis=0)
        except:
            XpTs_test = dict_scaled_data[COND][i]['XpT']
            XfTs_test = dict_scaled_data[COND][i]['XfT']
            YpTs_test = dict_scaled_data[COND][i]['YpT']
            YfTs_test = dict_scaled_data[COND][i]['YfT']
    dict_return = {'unscaled':dict_unscaled_data, 'scaled':dict_scaled_data, 'train':{}, 'valid':{}, 'test':{}}
    dict_return['train'] = {'XpTs': XpTs_train, 'XfTs': XfTs_train, 'YpTs': YpTs_train, 'YfTs': YfTs_train, 'indices': ls_train_indices}
    dict_return['valid'] = {'XpTs': XpTs_valid, 'XfTs': XfTs_valid, 'YpTs': YpTs_valid, 'YfTs': YfTs_valid, 'indices': ls_valid_indices}
    dict_return['test'] = {'XpTs': XpTs_test, 'XfTs': XfTs_test, 'YpTs': YpTs_test, 'YfTs': YfTs_test, 'indices': ls_test_indices}
    dict_return['X_scaler'] = X_scaler
    dict_return['Y_scaler'] = Y_scaler
    return dict_return

def resolve_complex_right_eigenvalues(E, W):
    eval = copy.deepcopy(np.diag(E))
    comp_modes = []
    comp_modes_conj = []
    for i1 in range(E.shape[0]):
        if np.imag(E[i1, i1]) != 0:
            print(i1)
            # Find the complex conjugate
            for i2 in range(i1 + 1, E.shape[0]):
                if eval[i2] == eval[i1].conj():
                    break
            # i1 and i2 are the indices of the complex conjugate eigenvalues
            comp_modes.append(i1)
            comp_modes_conj.append(i2)
            E[i1, i1] = np.real(eval[i1])
            E[i2, i2] = np.real(eval[i1])
            E[i1, i2] = np.imag(eval[i1])
            E[i2, i1] = - np.imag(eval[i1])
            u1 = copy.deepcopy(np.real(W[:, i1:i1 + 1]))
            w1 = copy.deepcopy(np.imag(W[:, i1:i1 + 1]))
            W[:, i1:i1 + 1] = u1
            W[:, i2:i2 + 1] = w1
    E_out = np.real(E)
    W_out = np.real(W)
    return E_out, W_out, comp_modes, comp_modes_conj

# ====================================================================================================================
# Gene Filtering Functions
# ====================================================================================================================
def filter_gene_by_coefficient_of_variation(dict_GrowthCurve, CV_THRESHOLD = np.Inf, MEAN_TPM_THRESHOLD = -np.Inf, ALL_CONDITIONS = ['MX','MN','NC']):
    # Get the gene locustag list
    temp_cond = list(dict_GrowthCurve.keys())[0]
    temp_curve = list(dict_GrowthCurve[temp_cond].keys())[0]
    ls_GENE_NAME = list(dict_GrowthCurve[temp_cond][temp_curve]['df_X_TPM'].index)
    ls_num_time_points = []
    for COND in ALL_CONDITIONS:
        ls_num_time_points.append(len(dict_GrowthCurve[COND][temp_curve]['df_X_TPM'].T))
    print('---------------------------------------------------')
    print('FILTER - COEFFICIENT OF VARIATION [INVERSE OF SNR]')
    # Get the coefficient of variation
    temp_np = np.empty(shape=(0, MAX_REPLICATES*MAX_LANES*MAX_READS))
    for COND_NO in range(len(ALL_CONDITIONS)):
        COND = ALL_CONDITIONS[COND_NO]
        temp_curve_data = np.empty(shape=(len(ls_GENE_NAME)*ls_num_time_points[COND_NO], 0))
        for CURVE in dict_GrowthCurve[COND]:
            temp_curve_data = np.concatenate([temp_curve_data, np.array(dict_GrowthCurve[COND][CURVE]['df_X_TPM']).T.reshape((-1,1))], axis=1)
        temp_np = np.concatenate([temp_np,temp_curve_data],axis=0)
    # temp_CV = np.array(pd.DataFrame(((temp_np.std(axis=1) / temp_np.mean(axis=1)).reshape((-1,len(ls_GENE_NAME))).T)).fillna(np.inf))
    temp_CV = np.array(pd.DataFrame(((temp_np.std(axis=1) / np.abs(temp_np.mean(axis=1))).reshape((-1, len(ls_GENE_NAME))).T)))
    # temp_CV_check_to_reject_genes = np.sum(temp_CV  > CV_THRESHOLD,axis=1) == np.sum(ls_num_time_points)
    # CV_THRESHOLD = 0.0125
    # print('Old filter rejects: ',np.sum(np.sum(temp_CV1 > CV_THRESHOLD,axis=1) == np.sum(ls_num_time_points)),' genes')
    temp_CV_check_to_reject_genes = []
    for i in range(len(temp_CV)):
        non_nan_vals = temp_CV[i][~np.isnan(temp_CV[i])]
        # print(non_nan_vals)
        if len(non_nan_vals) ==0:
            print('zero genes detected')
            temp_CV_check_to_reject_genes.append(True)
        elif (np.all(non_nan_vals >= CV_THRESHOLD)):
            temp_CV_check_to_reject_genes.append(True)
        else:
            temp_CV_check_to_reject_genes.append(False)
    # print('New filter rejects: ',np.sum(temp_CV_check_to_reject_genes), ' genes')
    ls_GENE_REMOVE1 = [ls_GENE_NAME[i] for i in range(len(ls_GENE_NAME)) if temp_CV_check_to_reject_genes[i] == True]
    ls_GENE_ALLOW1 = list(set(ls_GENE_NAME) - set(ls_GENE_REMOVE1))
    # temp_np = np.empty(shape=(len(ls_GENE_NAME), 0))
    # for COND in ALL_CONDITIONS:
    #     for CURVE in dict_GrowthCurve[COND]:
    #         temp_np = np.concatenate([temp_np,np.array(dict_GrowthCurve[COND][CURVE]['df_X_TPM'])],axis=1)
    # temp_CV = temp_np.std(axis=1)/temp_np.mean(axis=1)
    # ls_GENE_ALLOW = [ls_GENE_NAME[i] for i in range(len(ls_GENE_NAME)) if temp_CV[i]<CV_THRESHOLD]
    # ls_GENE_REMOVE = list(set(ls_GENE_NAME) - set(ls_GENE_ALLOW))
    print('The number of removed genes:',len(ls_GENE_REMOVE1))
    print('Remaining Genes:',len(ls_GENE_ALLOW1))
    print('---------------------------------------------------')
    print('FILTER - MEAN ')
    temp_mean = np.array(pd.DataFrame(temp_np.mean(axis=1).reshape((-1,len(ls_GENE_NAME))).T).fillna(0))
    temp_mean_check = np.sum(temp_mean < MEAN_TPM_THRESHOLD,axis=1) == np.sum(ls_num_time_points)
    ls_GENE_REMOVE2 = [ls_GENE_NAME[i] for i in range(len(ls_GENE_NAME)) if temp_mean_check[i] == True]
    ls_GENE_REMOVE2 = list(set(ls_GENE_REMOVE2) - set(ls_GENE_REMOVE1))
    ls_GENE_ALLOW2 = list(set(ls_GENE_ALLOW1) - set(ls_GENE_REMOVE2))
    # for items in ls_GENE_REMOVE2:
    #     print(items)
    print('The number of removed genes:', len(ls_GENE_REMOVE2))
    print('Remaining Genes:', len(ls_GENE_ALLOW2))
    print('---------------------------------------------------')
    for COND in ['MX','MN','NC']:
        for CURVE in dict_GrowthCurve[COND]:
            dict_GrowthCurve[COND][CURVE]['df_X_TPM'] = dict_GrowthCurve[COND][CURVE]['df_X_TPM'].drop(ls_GENE_REMOVE1,axis=0)
            dict_GrowthCurve[COND][CURVE]['df_X_TPM'] = dict_GrowthCurve[COND][CURVE]['df_X_TPM'].drop(ls_GENE_REMOVE2,axis=0)
    return dict_GrowthCurve



def denoise_using_PCA(dict_IN,PCA_THRESHOLD = 99,NORMALIZE = False,PLOT_SCREE=False):
    # Denoising the data across each time point using all the
    ls_curves = list(dict_IN.keys())
    n_genes = dict_IN[ls_curves[0]]['df_X_TPM'].shape[0]
    n_outputs = dict_IN[ls_curves[0]]['Y'].shape[0]
    gene_list = dict_IN[ls_curves[0]]['df_X_TPM'].index
    ls_time_pts = list(dict_IN[ls_curves[0]]['df_X_TPM'].columns)
    dict_OUT = {}
    # PCA to denoise the Growth curve outputs
    Y = np.empty(shape=(n_outputs * len(ls_time_pts), 0))
    for curve in ls_curves:
        Y = np.concatenate([Y, np.array(dict_IN[curve]['Y']).T.reshape(-1, 1)], axis=1)
    Uy, Sy, VyT = np.linalg.svd(Y)
    if PLOT_SCREE:
        plt.stem(np.concatenate([np.array([100]), (1 - np.cumsum(Sy ** 2) / np.sum(Sy ** 2)) * 100], axis=0))
        plt.ylabel('Uncaptured % of signal')
        plt.xlabel('Number of Principal Components')
        plt.title('Scree plot of all outputs')
        plt.show()
    Yhat = np.matmul(Uy[:, 0:1], VyT[0:1, :]) * Sy[0]
    for i in range(len(ls_curves)):
        dict_OUT[ls_curves[i]] = {'df_X_TPM':{},'Y':pd.DataFrame(Yhat[:,i].reshape(-1,n_outputs).T,columns=ls_time_pts,index = ['y' + str(i+1) for i in range(n_outputs)])}

    # Making all entries nonzero
    for curve in ls_curves:
        dict_IN[curve]['df_X_TPM'] = dict_IN[curve]['df_X_TPM'].replace(0,1e-5)
    # PCA to denoise the states
    for time_point in range(1,8):
        X = np.empty(shape=(n_genes, 0))
        for curve in ls_curves:
            X = np.concatenate([X,np.array(dict_IN[curve]['df_X_TPM'].loc[:,time_point]).reshape(-1,1)],axis=1)
        if NORMALIZE:
            # Normalize the data
            X_mu = np.mean(X,axis=1).reshape(-1,1)
            X_sigma = np.std(X, axis=1).reshape(-1, 1)
            for i in range(X_sigma.shape[0]):
                if X_sigma[i][0] == 0:
                    X_sigma[i][0] = 1e-10
        else:
            X_mu = 0
            X_sigma = 1
        X_norm = (X - X_mu)/X_sigma
        # SVD of all the data
        U,S,VT = np.linalg.svd(X_norm)
        if PLOT_SCREE:
            # Plot the SVD scree plot
            plt.stem(np.concatenate([np.array([100]), (1-np.cumsum(S**2)/np.sum(S**2))*100],axis=0))
            plt.ylabel('Uncaptured % of signal')
            plt.xlabel('Number of Principal Components')
            plt.title('Scree plot of Timepoint - ' + str(time_point))
            plt.show()
        # Data Reconstruction and storage
        nPC_opt = np.nonzero(S * (np.cumsum(S ** 2) / np.sum(S ** 2) * 100 > PCA_THRESHOLD))[0][0]+1
        X_norm_hat = np.matmul(U[:,0:nPC_opt],np.matmul(np.linalg.inv(np.diag(S[0:nPC_opt])),VT[0:nPC_opt,:]))
        if PLOT_SCREE:
            # Plot the error in each gene across time
            SSE = np.sum(((X_norm_hat*X_sigma + X_mu) - X)**2,axis=1)
            SST = np.sum(X**2,axis=1)
            r2 = np.maximum(0,(1-SSE/SST)*100)
            plt.figure(figsize=(10,2))
            plt.plot(r2)
            plt.ylabel('$r^2$')
            plt.xlabel('Gene_Locus Tag')
            plt.title('Reconstruction Error of Timepoint - ' + str(time_point))
            plt.show()
        for curve in ls_curves:
            df_temp = pd.DataFrame(X_norm_hat[:, curve:(curve + 1)]*X_sigma +X_mu, index=gene_list, columns=[time_point])
            if time_point == 1:
                dict_OUT[curve]['df_X_TPM'] = copy.deepcopy(df_temp)
            else:
                dict_OUT[curve]['df_X_TPM'] = pd.concat([dict_OUT[curve]['df_X_TPM'],copy.deepcopy(df_temp)],axis=1)

    # Coefficient of variation of all the genes
    n_curves = len(ls_curves)
    Xtrue = np.empty(shape=(n_genes, n_curves, len(ls_time_pts)))
    Xhat = np.empty(shape=(n_genes, n_curves, len(ls_time_pts)))
    for curve, time_pt in itertools.product(range(n_curves), ls_time_pts):
        Xtrue[:, curve:(curve + 1), (time_pt - 1):time_pt] = np.array(dict_IN[curve]['df_X_TPM'].loc[:, time_pt]).reshape(-1, 1, 1)
        Xhat[:, curve:(curve + 1), (time_pt - 1):time_pt] = np.array(dict_OUT[curve]['df_X_TPM'].loc[:, time_pt]).reshape(-1, 1, 1)
    #
    plt.figure(figsize=(10, 6))
    for time_pt in range(len(ls_time_pts)):
        X_i_hat = copy.deepcopy(Xhat[:, :, time_pt]).reshape(n_genes, -1)
        plt.plot(np.mean(X_i_hat, axis=1) / np.std(X_i_hat, axis=1),label='Timepoint ' + str(time_pt + 1))  # , color='green')
    for time_pt in range(len(ls_time_pts)):
        X_i_true = copy.deepcopy(Xtrue[:, :, time_pt]).reshape(n_genes, -1)
        if time_pt == 1:
            plt.plot(np.mean(X_i_true, axis=1) / np.std(X_i_true, axis=1), '.', color='blue',label = 'All timepoints \n before filtering')
        else:
            plt.plot(np.mean(X_i_true, axis=1) / np.std(X_i_true, axis=1), '.', color='blue')
    plt.legend()
    plt.show()
    print('Denoising using PCA is complete')
    return dict_OUT

def get_gene_conversion_info():
    RNA_FILE_LOCATION = '/Users/shara/Desktop/oc_deepDMD/DATA/RNA_1_Pput_R2A_Cas_Glu/raw_rnaseq_files'
    ls_RNA_files = os.listdir(RNA_FILE_LOCATION)
    df_temp = pd.read_csv(RNA_FILE_LOCATION + '/' + ls_RNA_files[0])
    df_temp.index = df_temp.locus_tag
    df_temp = df_temp.loc[:,['Name', 'protein_id', 'gene']]
    return df_temp

from bioservices import UniProt
# ====================================================================================================================
# Functions to query Uniprot
# ====================================================================================================================
# The file format is ALWAYS set to tab delimited
# For the valid column names refer to the website: https://www.uniprot.org/help/uniprotkb_column_names



def get_gene_Uniprot_DATA(species_id='KT2440', ls_all_locus_tags='PP_0123', search_columns='entry name,length,id, genes,comment(FUNCTION)'):
    query_search = '\"' + species_id + '\" AND ('
    for locus_tag in ls_all_locus_tags[0:-1]:
        query_search = query_search + '\"' + locus_tag + '\" OR '
    query_search = query_search + '\"' + ls_all_locus_tags[-1] + '\")'
    up = UniProt()
    search_result = up.search(query_search, frmt='tab',columns=search_columns)
    # Creating the dataframe with the obtained entries
    # up - uniprot
    str_up_ALL = search_result.split('\n')
    ls_up = []
    for each_line in str_up_ALL[1:]:
        ls_up.append(each_line.split('\t'))
    df_up = pd.DataFrame(ls_up[0:-1])
    df_up.columns = str_up_ALL[0].split('\t')
    return df_up

def get_Uniprot_cell_division_genes_and_cell_cycle_genes(species_id='KT2440', search_columns='genes(OLN), genes(PREFERRED), go(biological process)'):
    query_search = {'cell cycle':'cell goa:(cycle) ' + species_id, 'cell division':'cell goa:(division) ' + species_id}
    up = UniProt()
    dict_out ={}
    for items in query_search.keys():
        search_result_i = up.search(query_search[items], frmt='tab', columns=search_columns)
        str_up_ALL = search_result_i.split('\n')
        ls_up = []
        for each_line in str_up_ALL[1:]:
            ls_up.append(each_line.split('\t'))
        df_up = pd.DataFrame(ls_up[0:-1])
        df_up.columns = str_up_ALL[0].split('\t')
        dict_out[items] = copy.deepcopy(df_up)
    set_genes1 = set(dict_out['cell cycle'].iloc[:, 0].unique())
    set_genes2 = set(dict_out['cell division'].iloc[:, 0].unique())
    ls_out = set_genes1.union(set_genes2) - {''}
    return dict_out,ls_out

# def get_Uniprot_cell_division_genes_and_cell_cycle_genes(species_id='KT2440', search_columns='genes(OLN), genes(PREFERRED), go(biological process)'):
#     dict_gene_ontology_biological_processes = {}
#     dict_gene_ontology_biological_processes['GO:0009083'] = 'branched-chain amino acid catabolic process'
#     dict_gene_ontology_biological_processes['GO:0042219'] = 'cellular modified amino acid catabolic process'
#     dict_gene_ontology_biological_processes['GO:0006575'] = 'cellular modified amino acid metabolic process'
#     dict_gene_ontology_biological_processes['GO:0008652'] = 'cellular amino acid biosynthetic process'
#     dict_gene_ontology_biological_processes['GO:0009063'] = 'cellular amino acid catabolic process'
#     dict_gene_ontology_biological_processes['GO:0006520'] = 'cellular amino acid metabolic process'
#     dict_gene_ontology_biological_processes['GO:0006865'] = 'amino acid transport'
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#     dict_gene_ontology_biological_processes[''] = ''
#
#
#     query_search = {'cell cycle':'cell goa:(cycle) ' + species_id, 'cell division':'cell goa:(division) ' + species_id}
#     up = UniProt()
#     dict_out ={}
#     for items in query_search.keys():
#         search_result_i = up.search(query_search[items], frmt='tab', columns=search_columns)
#         str_up_ALL = search_result_i.split('\n')
#         ls_up = []
#         for each_line in str_up_ALL[1:]:
#             ls_up.append(each_line.split('\t'))
#         df_up = pd.DataFrame(ls_up[0:-1])
#         df_up.columns = str_up_ALL[0].split('\t')
#         dict_out[items] = copy.deepcopy(df_up)
#     set_genes1 = set(dict_out['cell cycle'].iloc[:, 0].unique())
#     set_genes2 = set(dict_out['cell division'].iloc[:, 0].unique())
#     ls_out = set_genes1.union(set_genes2) - {''}
#     return dict_out,ls_out



# SUB OPTIMAL CODE
# def get_gene_Uniprot_DATA(species_id='KT2440', ls_all_locus_tags='PP_0123', search_columns='entry name,length,id, genes,comment(FUNCTION)'):
#     for locus_tag in ls_all_locus_tags:#ls_all_locus_tags[0:-1]:
#         query_search = locus_tag + ' AND ' + species_id
#         up = UniProt()
#         search_result = up.search(query_search, frmt='tab',columns=search_columns)
#
#         # Creating the dataframe with the obtained entries
#         # up - uniprot
#         str_up_ALL = search_result.split('\n')
#         ls_up = []
#         for each_line in str_up_ALL[1:]:
#             ls_up.append(each_line.split('\t'))
#         df_inter = pd.DataFrame(ls_up[0:-1])
#         df_inter.columns = str_up_ALL[0].split('\t')
#         try:
#             df_up = pd.concat([df_up,df_inter],axis=0)
#         except:
#             df_up = copy.deepcopy(df_inter)
#     return df_up


# def get_gene_Uniprot_DATA_old(species_id='KT2440', ls_all_locus_tags='PP_0123',
#                           search_columns='entry name,length,id, genes,comment(FUNCTION)'):
#     query_search =''
#     for locus_tag in ls_all_locus_tags[0:-1]:
#         query_search = query_search + locus_tag + ' OR '
#     query_search = query_search + ls_all_locus_tags[-1] + ' AND ' + species_id
#     up = UniProt()
#     search_result = up.search(query_search, frmt='tab',columns=search_columns)
#
#     # Creating the dataframe with the obtained entries
#     # up - uniprot
#     str_up_ALL = search_result.split('\n')
#     ls_up = []
#     for each_line in str_up_ALL[1:]:
#         ls_up.append(each_line.split('\t'))
#     df_up = pd.DataFrame(ls_up[0:-1])
#     df_up.columns = str_up_ALL[0].split('\t')
#     return df_up



# ====================================================================================================================
# Prediction Functions
# ====================================================================================================================

def generate_n_step_prediction_table(SYSTEM_NO,ALL_CONDITIONS=['MX'],ls_runs1=list(range(0,100)),METHOD = 'Sequential'):
    ocdeepDMD_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle'
    original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
    indices_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_OrderedIndices.pickle'
    root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    dict_predict_STATS_file = root_run_file + '/dict_predict_STATS.pickle'
    # Indices [train, validation and test]
    with open(indices_path, 'rb') as handle:
        ls_data_indices = pickle.load(handle)
    ls_train_indices = ls_data_indices[0:12]
    ls_valid_indices = ls_data_indices[12:14]
    ls_test_indices = ls_data_indices[14:16]
    # Datasets [sorted as scaled and unscaled] and Conditions
    with open(original_data_path, 'rb') as handle:
        dict_data_original = pickle.load(handle)

    n_genes = len(dict_data_original[ALL_CONDITIONS[0]][ls_data_indices[0]]['df_X_TPM'])

    dict_empty_all_conditions = {}
    for COND in ALL_CONDITIONS:
        dict_empty_all_conditions[COND] = {}

    dict_temp = get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS=ALL_CONDITIONS)
    dict_scaled_data = dict_temp['scaled']
    dict_unscaled_data = dict_temp['unscaled']

    # Generate predictions for each curve and write down the error statistics for each run
    ls_all_run_indices = []
    for folder in os.listdir(root_run_file + '/' + METHOD):
        if folder[0:4] == 'RUN_':  # It is a RUN folder
            ls_all_run_indices.append(int(folder[4:]))
    ls_runs1 = set(ls_runs1).intersection(set(ls_all_run_indices))
    # Open the predictions folder or create one if it doesn't exist
    try:
        with open(dict_predict_STATS_file, 'rb') as handle:
            dict_predict_STATS = pickle.load(handle)
    except:
        dict_predict_STATS = {}

    dict_resultable1 = {}
    # Generate the predictions for each run
    for run in ls_runs1:
        dict_resultable1[run] = {}
        print('RUN: ', run)
        sess = tf.InteractiveSession()
        run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
        saver = tf.compat.v1.train.import_meta_graph(
            run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
        saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
        dict_params = {}
        dict_params['psixpT'] = tf.get_collection('psixpT')[0]
        dict_params['psixfT'] = tf.get_collection('psixfT')[0]
        dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
        dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
        dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])

        dict_instant_run_result = copy.deepcopy(dict_empty_all_conditions)
        for items in dict_instant_run_result.keys():
            dict_instant_run_result[items] = {'train_Xf_1step': [], 'train_Xf_nstep': [], 'valid_Xf_1step': [],
                                              'valid_Xf_nstep': [], 'test_Xf_1step': [], 'test_Xf_nstep': []}
        for COND, data_index in itertools.product(ALL_CONDITIONS, ls_data_indices):
            # Figure out if the index belongs to train, test or validation
            if data_index in ls_train_indices:
                key2_start = 'train_'
            elif data_index in ls_valid_indices:
                key2_start = 'valid_'
            elif data_index in ls_test_indices:
                key2_start = 'test_'
            # else:
            #     print('ERROR!!!')
            #     return 0
            # --- *** Generate prediction *** ---

            # Xf - 1 step
            psiXpT = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: dict_scaled_data[COND][data_index]['XpT']})
            psiXfT_hat = np.matmul(psiXpT, dict_params['KxT_num'])
            XfT_hat = dict_temp['X_scaler'].inverse_transform(psiXfT_hat[:, 0:n_genes])
            dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfT_hat))#, multioutput='variance_weighted'))
            # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(np.mean(np.square(XfT_hat)))
            # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfT_hat[:,0:n_genes]))#, multioutput='variance_weighted'))
            # dict_instant_run_result[COND][key2_start + 'Xf_1step'].append(r2_score(dict_params['psixfT'].eval(
            #     feed_dict={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), psiXfT_hat, multioutput='variance_weighted'))

            # Xf - n step
            psiXfTn_hat = psiXpT[0:1, :]  # get the initial condition
            for i in range(len(dict_scaled_data[COND][data_index]['XfT'])):  # predict n - steps
                psiXfTn_hat = np.concatenate([psiXfTn_hat, np.matmul(psiXfTn_hat[-1:], dict_params['KxT_num'])], axis=0)
            psiXfTn_hat = psiXfTn_hat[1:, :]
            # Remove the initial condition and the lifted states; then unscale the variables
            XfTn_hat = dict_temp['X_scaler'].inverse_transform(psiXfTn_hat[:, 0:n_genes])
            dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_unscaled_data[COND][data_index]['XfT'], XfTn_hat))#, multioutput='variance_weighted'))
            # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(np.mean(np.square(XfTn_hat)))
            # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_scaled_data[COND][data_index]['XfT'], psiXfTn_hat[:, 0:n_genes],multioutput='variance_weighted'))
            # dict_instant_run_result[COND][key2_start + 'Xf_nstep'].append(r2_score(dict_params['psixfT'].eval(
            #     feed_dict={dict_params['xfT_feed']: dict_scaled_data[COND][data_index]['XfT']}), psiXfTn_hat, multioutput='variance_weighted'))

            # --- *** Compute the stats *** --- [for training, validation and test data sets separately]
        # Save the stats to the dictionary - for MX,MN and NC, we save (train, test, valid) * (Xf1step, Xfnstep, Yf1step, Yfnstep)
        for COND in dict_instant_run_result.keys():
            for items in dict_instant_run_result[COND].keys():
                dict_instant_run_result[COND][items] = np.mean(dict_instant_run_result[COND][items])
        dict_predict_STATS[run] = pd.DataFrame(dict_instant_run_result).T
        dict_resultable1[run]['train_Xf_1step'] = dict_predict_STATS[run].loc[:, 'train_Xf_1step'].mean()
        dict_resultable1[run]['valid_Xf_1step'] = dict_predict_STATS[run].loc[:, 'valid_Xf_1step'].mean()
        dict_resultable1[run]['test_Xf_1step'] = dict_predict_STATS[run].loc[:, 'test_Xf_1step'].mean()
        dict_resultable1[run]['train_Xf_nstep'] = dict_predict_STATS[run].loc[:, 'train_Xf_nstep'].mean()
        dict_resultable1[run]['valid_Xf_nstep'] = dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean()
        dict_resultable1[run]['test_Xf_nstep'] = dict_predict_STATS[run].loc[:, 'test_Xf_nstep'].mean()
        tf.reset_default_graph()
        sess.close()

    print('============================================================================')
    print('RESULT TABLE 1')
    df_resultable1 = pd.DataFrame(dict_resultable1).T
    print(df_resultable1)
    print('============================================================================')

    # Need a plot of the results

    dict_resultable_2 = {}
    for run in dict_predict_STATS.keys():
        with open(root_run_file + '/' + METHOD + '/Run_' + str(run) + '/dict_hyperparameters.pickle', 'rb') as handle:
            dict_hp = pickle.load(handle)
        dict_resultable_2[run] = {'x_obs': dict_hp['x_obs'],
                                  'n_l & n_n': [dict_hp['x_layers'], dict_hp['x_nodes']], 'r2_X_nstep_train':
                                      dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(), 'r2_X_nstep_valid':
                                      dict_predict_STATS[run].loc[:, 'valid_Xf_nstep'].mean(), 'r2_X_nstep_test':
                                      dict_predict_STATS[run].loc[:, 'test_Xf_nstep'].mean(),
                                  'lambda': dict_hp['regularization factor']}
        # dict_resultable_2[run] = {'x_obs': dict_hp['x_obs'],
        #                           'n_l & n_n': [dict_hp['x_layers'], dict_hp['x_nodes']], 'r2_X_1step_train':
        #                               dict_predict_STATS[run].loc[:, 'valid_Xf_1step'].mean(), 'r2_X_1step_valid':
        #                               dict_predict_STATS[run].loc[:, 'valid_Xf_1step'].mean(), 'r2_X_1step_test':
        #                               dict_predict_STATS[run].loc[:, 'test_Xf_1step'].mean()}
    df_resultable2 = pd.DataFrame(dict_resultable_2).T.sort_values(by='x_obs')
    print('============================================================================')
    print('RESULT TABLE 2')
    print(df_resultable2)
    print('============================================================================')
    return df_resultable2


def plot_dynamics_related_graphs(SYSTEM_NO,run,METHOD,ALL_CONDITIONS=['MX']):
    root_run_file = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    original_data_path = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(
        SYSTEM_NO) + '/System_' + str(SYSTEM_NO) + '_Data.pickle'
    with open(original_data_path, 'rb') as handle:
        dict_data_original = pickle.load(handle)
    sess = tf.InteractiveSession()
    run_folder_name = root_run_file + '/' + METHOD + '/RUN_' + str(run)
    saver = tf.compat.v1.train.import_meta_graph(
        run_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle.ckpt.meta', clear_devices=True)
    saver.restore(sess, tf.train.latest_checkpoint(run_folder_name))
    dict_params = {}
    dict_params['psixpT'] = tf.get_collection('psixpT')[0]
    dict_params['psixfT'] = tf.get_collection('psixfT')[0]
    dict_params['xpT_feed'] = tf.get_collection('xpT_feed')[0]
    dict_params['xfT_feed'] = tf.get_collection('xfT_feed')[0]
    dict_params['KxT_num'] = sess.run(tf.get_collection('KxT')[0])
    Kx = dict_params['KxT_num'].T
    e_in, W_in = np.linalg.eig(Kx)
    E_in = np.diag(e_in)
    E, W, comp_modes, comp_modes_conj = resolve_complex_right_eigenvalues(copy.deepcopy(E_in), copy.deepcopy(W_in))
    Winv = np.linalg.inv(W)

    # Plot for the K
    Kx = dict_params['KxT_num'].T
    E_complex = np.linalg.eigvals(Kx)
    # K matrix heatmap
    plt.figure(figsize=(12, 10))
    a = sb.heatmap(Kx, cmap="RdYlGn", center=0, vmax=np.abs(Kx).max(), vmin=-np.abs(Kx).max())
    b, t = a.axes.get_ylim()  # discover the values for bottom and top
    b += 0.5  # Add 0.5 to the bottom
    t -= 0.5  # Subtract 0.5 from the top
    cbar = a.collections[0].colorbar
    ls_gene_tags = list(dict_data_original['MX'][0]['df_X_TPM'].index)
    p = get_gene_Uniprot_DATA(ls_all_locus_tags=ls_gene_tags, search_columns='genes(OLN),genes(PREFERRED)')
    ls_genes = []
    for i in range(len(ls_gene_tags)):
        if p[p['Gene names  (ordered locus )'] == ls_gene_tags[i]].iloc[0, 1] == '':
            ls_genes.append(p[p['Gene names  (ordered locus )'] == ls_gene_tags[i]].iloc[0, 0])
        else:
            ls_genes.append(p[p['Gene names  (ordered locus )'] == ls_gene_tags[i]].iloc[0, 1])
    for i in range(len(Kx) - len(ls_genes) - 1):
        ls_genes.append('$\\varphi_{{{}}}(x)$'.format(i + 1))
    ls_genes.append('$\\varphi_{0}(x)$')
    a.set_xticks(np.arange(0.5, len(ls_genes), 1))
    a.set_yticks(np.arange(0.5, len(ls_genes), 1))
    a.set_xticklabels(ls_genes, rotation=90, fontsize=19)
    a.set_yticklabels(ls_genes, rotation=0, fontsize=19)
    a.axes.set_ylim(b, t)
    # here set the labelsize by 20
    # cbar.ax.tick_params(labelsize=FONTSIZE)
    # a.axes.set_xticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation=90)
    # a.axes.set_yticklabels(ls_gene_names,{'fontsize':FONTSIZE},rotation = 0)
    plt.show()

    # Eigenvalue plot
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(1, 1, 1)
    circ = plt.Circle((0, 0), radius=1, edgecolor='None', facecolor='cyan')
    ax.add_patch(circ)
    ax.plot(np.real(E_complex), np.imag(E_complex), 'x', linewidth=5, color='g', markersize=12)
    ax.set_xlabel('$Re(\lambda)$')
    ax.set_ylabel('$Im(\lambda)$')
    ax.set_xticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    plt.show()

    # Eigenfunction plot
    dict_ALLDATA = get_train_test_valid_data(SYSTEM_NO, ALL_CONDITIONS=ALL_CONDITIONS)
    index = 0
    n_funcs = len(E) - len(comp_modes)
    n_funcs = n_funcs + np.mod(n_funcs,2)
    XT = np.concatenate(
        [dict_ALLDATA['unscaled']['MX'][index]['XpT'][0:1, :], dict_ALLDATA['unscaled']['MX'][index]['XfT']], axis=0)
    XTs = dict_ALLDATA['X_scaler'].transform(XT)
    psiXTs_true = dict_params['psixpT'].eval(feed_dict={dict_params['xpT_feed']: XTs})
    psiXTs = dict_params['psixpT'].eval(
        feed_dict={dict_params['xpT_feed']: dict_ALLDATA['scaled']['MX'][index]['XpT'][0:1, :]})
    for i in range(len(dict_ALLDATA['unscaled']['MX'][index]['XfT'])):
        psiXTs = np.concatenate([psiXTs, np.matmul(psiXTs[-1:], dict_params['KxT_num'])])
    Phis = np.matmul(Winv, psiXTs.T)
    # Phis = np.matmul(Winv, psiXTs_true.T)
    YT = np.concatenate(
        [dict_ALLDATA['unscaled']['MX'][index]['YpT'][0:1, :], dict_ALLDATA['unscaled']['MX'][index]['YfT']], axis=0)
    YTs = dict_ALLDATA['Y_scaler'].transform(YT)

    x_ticks = np.array([1, 2, 3, 4, 5, 6, 7])
    f, ax_o = plt.subplots(np.int(np.ceil(n_funcs / 2)), 2, sharex=True, sharey= True,figsize=(10, n_funcs * 1.5))
    # f,ax = plt.subplots(n_funcs,1,sharex=True,figsize=(5,n_funcs*1.5))
    ax = ax_o.reshape(-1)
    eig_func_index = 0
    for i in range(n_funcs):
        try:
            if eig_func_index in comp_modes:
                ax[i].plot(x_ticks, Phis[eig_func_index, :]/np.max(np.abs(Phis[eig_func_index, :])), label='Real')
                ax[i].plot(x_ticks, Phis[eig_func_index + 1, :]/np.max(np.abs(Phis[eig_func_index + 1, :])), label='Imaginary')
                # ax[i].legend()
                real_part = round(np.abs(E[eig_func_index, eig_func_index]), 3)
                imag_part = round(np.abs(E[eig_func_index, eig_func_index + 1]), 3)
                ax[i].set_title(
                    '$\phi_{{{},{}}}(x)$'.format(eig_func_index + 1, eig_func_index + 2) + ', $\lambda =$' + str(
                        real_part) + '$\pm$j' + str(imag_part))
                eig_func_index = eig_func_index + 2
            else:
                ax[i].plot(x_ticks, Phis[eig_func_index, :]/np.max(np.abs(Phis[eig_func_index, :])),linewidth =2)
                real_part = round(np.abs(E[eig_func_index, eig_func_index]), 3)
                # print(real_part)
                # ax[i].set_title('$\phi_{{{}}}(x), \lambda = {{{}}}$'.format(eig_func_index+1,real_part))
                ax[i].set_title('$\phi_{{{}}}(x)$'.format(eig_func_index + 1) + ', $\lambda = $' + str(real_part))
                eig_func_index = eig_func_index + 1
            ax[i].set_ylim([-1, 1])
        except:
            break
    ax_o[-1, 0].set_xlabel('time (hrs)')
    ax_o[-1, 1].set_xlabel('time (hrs)')
    ax_o[-1, 0].set_xticks([0, 3, 6])
    ax_o[-1, 1].set_xticks([0, 3, 6])
    f.show()

    # Plot of the genes - base states
    ls_gene_max_var_index = sorted(range(len(XT.var(axis=0))), key=lambda i: XT.var(axis=0)[i])
    ls_gene_max_var_index.reverse()
    plt.figure(figsize=(10, 6))
    for i in range(n_funcs):
        try:
            plt.plot(x_ticks, XTs[:, ls_gene_max_var_index[i]], label='gene_' + str(i))
        except:
            break
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=22, ncol=4)
    plt.show()

    # Plot of the observables
    plt.figure(figsize=(10, 6))
    for i in range(len(psiXTs[0]) - len(XTs[0])):
        if i == len(psiXTs[0]):
            plt.plot(x_ticks, psiXTs[:, len(XTs[0]) + i], label='$\psi_{0}(x)$')
        else:
            plt.plot(x_ticks, psiXTs[:, len(XTs[0]) + i], label='$\psi_{{{}}}(x)$'.format(i + 1))
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), fontsize=22, ncol=4)
    plt.show()

    tf.reset_default_graph()
    sess.close()

    return

def save_best_run_of_Seq_OCdeepDMD_problem_1(SYSTEM_NO,RUN_NO):
    METHOD = 'Sequential'
    sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
    run_folder_name = sys_folder_name + '/' + METHOD + '/RUN_' + str(RUN_NO)
    with open(run_folder_name + '/constrainedNN-Model.pickle', 'rb') as handle:
        d = pickle.load(handle)
    with open(run_folder_name + '/dict_hyperparameters.pickle', 'rb') as handle:
        d1 = pickle.load(handle)
    for items in d1.keys():
        d[items] = d1[items]
    # print(d.keys())
    with open('/Users/shara/Desktop/oc_deepDMD/System_' + str(SYSTEM_NO) + '_BestRun_1.pickle', 'wb') as handle:
        pickle.dump(d, handle)
    return



