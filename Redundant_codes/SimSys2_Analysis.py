##
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import ocdeepdmd_simulation_examples_helper_functions as oc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import shutil

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


SYSTEM_NO = 7
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)

# Theoretical Eigenfunctions
# A = np.array([[0.86,0.],[0.8,0.4]])
a11 = 0.86
a21 = 0.8
a22 = 0.4
gamma = -0.9

A = np.asarray([[a11,0,0,0,0],[a21,a22,0,gamma,0],[0,0,a11*a22,a11*a21,a11*gamma],[0,0,0,a11**2,0],[0,0,0,0,a11*3]])