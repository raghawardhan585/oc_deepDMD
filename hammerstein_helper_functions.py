import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
import random
import os
import shutil
import tensorflow as tf
import copy
import itertools
from scipy.integrate import odeint
import ocdeepdmd_simulation_examples_helper_functions as oc
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def transfer_current_ocDeepDMD_run_files():
    runinfo_folder = '/Users/shara/Desktop/oc_deepDMD/Run_info'
    source_folder = '/Users/shara/Desktop/oc_deepDMD/_current_run_saved_files'
    # Find the SYSTEM NUMBER
    # Assumption: All the folders in the _current_run_saved_files belong to the same system
    for items in os.listdir(source_folder):
        if items[0:4] == 'SYS_':
            for i in range(4, len(items)):
                if items[i] == '_':
                    SYSTEM_NUMBER = int(items[4:i])
                    break
            break
    # Find HIGHEST RUN NUMBER
    destination_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NUMBER) + '/Hammerstein'
    if not os.path.exists(destination_folder):
        os.mkdir(destination_folder)
    current_run_no = -1
    ls_complete_runs = []
    for items in os.listdir(destination_folder):
        if items[0:4] == 'RUN_':
            current_run_no = np.max([current_run_no, int(items[4:])])
            ls_complete_runs.append(int(items[4:]))
    ls_missing_runs = list(set(range(current_run_no))-set(ls_complete_runs))
    # Transfer files to missing folders
    n_miss = len(ls_missing_runs)
    if n_miss>0:
        i = 0
        for items in list(set(os.listdir(source_folder)) - {'.DS_Store'}):
            shutil.move(source_folder + '/' + items, destination_folder + '/RUN_' + str(ls_missing_runs[i]))
            shutil.move(runinfo_folder + '/' + items + '.txt',
                        destination_folder + '/RUN_' + str(ls_missing_runs[i]) + '/RUN_' + str(ls_missing_runs[i]) + '.txt')
            i = i+1
            if i == n_miss:
                break
    # Transfer the files to new folders
    current_run_no = current_run_no + 1
    for items in list(set(os.listdir(source_folder)) - {'.DS_Store'}):
        shutil.move(source_folder + '/' + items, destination_folder + '/RUN_' + str(current_run_no))
        shutil.move(runinfo_folder + '/' + items + '.txt',
                    destination_folder + '/RUN_' + str(current_run_no) + '/RUN_' + str(current_run_no) + '.txt')
        current_run_no = current_run_no + 1
    return