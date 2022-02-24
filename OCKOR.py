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

plt.rcParams["font.family"] = "Times"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["font.size"] = 22



class OCKOR():
    def __init__(self,df_RNA,df_OD,df_U,data_start_index):
        # TODO - check that the metadata of df_RNA and the metadata of RNAseq are the same
        self.X = df_RNA # contains the metadata as well
        self.data_start_index = data_start_index
        self.Y = df_OD
        self.U = df_U








