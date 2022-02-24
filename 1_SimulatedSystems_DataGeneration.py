##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import pickle
import random
import os
import shutil
import tensorflow as tf

import ocdeepdmd_simulation_examples_helper_functions as oc
colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette

# ==========================
## System 1 Data Generation
# ==========================
# System Parameters
A = np.array([[0.86,0.],[0.8,0.4]])
gamma = 0
# Simulation Parameters
N_data_points = 30
N_CURVES = 10
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
SYSTEM_NO = 1
ls_prediction_steps=[1]
oc.data_gen_sys_1_2(sys_params, N_CURVES, SYSTEM_NO)

# ==========================
## System 2 Data Generation
# ==========================
# System Parameters
A = np.array([[0.9,0.],[-0.4,-0.8]])
gamma = -0.9
# Simulation Parameters
N_data_points = 20
N_CURVES = 300
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
SYSTEM_NO = 1
ls_prediction_steps=[1]
oc.data_gen_sys_1_2(sys_params, N_CURVES, SYSTEM_NO)
oc.plot_training_valid_test_states(SYSTEM_NO)
# ##
# with open('koopman_data/System_4_ocdeepDMDdata.pickle','rb') as handle:
#     a = pickle.load(handle)
# plt.figure()
# plt.plot(a['Yp'])
# plt.plot(a['Xp'])
# plt.show()

# ==========================
## System 3 - Activator Repressor CLock - 4state system
# ==========================
# System Parameters
gamma_A = 1.
gamma_B = 0.5
delta_A = 1.
delta_B = 1.
alpha_A0= 0.04
alpha_B0= 0.004
alpha_A = 50.
alpha_B = 30.
K_A = 1.
K_B = 1.5
kappa_A = 1.
kappa_B = 1.
n = 2.
m = 2.
k_3n = 3.
k_3d = 1.08

# gamma_A = 1.
# gamma_B = 0.5
# delta_A = 1.
# delta_B = 1.
# alpha_A0= 0.04
# alpha_B0= 0.004
# alpha_A = 250.
# alpha_B = 30.
# K_A = 1.
# K_B = 1.
# kappa_A = 1.
# kappa_B = 1.
# n = 2.
# m = 4.
# k_3n = 3.
# k_3d = 1.08

# x_min = np.asarray([0.1,0.3,0.9,0.3])
# x_max = np.asarray([0.2,1.,1.,1.])
x_min = np.asarray([0.3,0.3,0.9,1.3])
# x_max = np.asarray([0.4,0.4,0.95,1.4])
x_max = x_min

sys_params_arc4s = (gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m)
# Simulation Parameters
sampling_time = 0.5
simulation_time = 100
N_CURVES = 3
# ls_prediction_steps=[1]#list(range(1,10,1))

sys_params = {'sys_params_arc4s': sys_params_arc4s , 'k_3n':k_3n, 'k_3d':k_3d, 'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES, 'x_min': x_min, 'x_max':x_max}
SYSTEM_NO = 22
oc.data_gen_sys_arc4s(sys_params, N_CURVES,SYSTEM_NO)
oc.plot_one_curve(SYSTEM_NO,CURVE_NO=0)
oc.plot_training_valid_test_states(SYSTEM_NO)


# ==========================
## System 3b - Activator Repressor CLock - 2state system
# ==========================
# System Parameters
gamma_A = 1.
gamma_B = 0.5
delta_A = 1.
delta_B = 1.
alpha_A0= 0.04
alpha_B0= 0.004
alpha_A = 50.
alpha_B = 30.
K_A = 1.
K_B = 1.5
kappa_A = 1.
kappa_B = 1.
n = 2.
m = 2.
k_3n = 1.
k_3d = 0.6
cooperativity_1 = 1
cooperativity_2 = 3

# gamma_A = 1.
# gamma_B = 0.5
# delta_A = 1.
# delta_B = 1.
# alpha_A0= 0.04
# alpha_B0= 0.004
# alpha_A = 250.
# alpha_B = 30.
# K_A = 1.
# K_B = 1.
# kappa_A = 1.
# kappa_B = 1.
# n = 2.
# m = 4.
# k_3n = 3.
# k_3d = 1.08

x_min = np.asarray([1,1])
x_max = np.asarray([25,70])

sys_params_arc2s = (gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m)
# Simulation Parameters
sampling_time = 0.5
simulation_time = 50
N_CURVES = 300

sys_params = {'sys_params_arc2s': sys_params_arc2s , 'k_3n':k_3n, 'k_3d':k_3d, 'cooperativity_1':cooperativity_1, 'cooperativity_2': cooperativity_2, 'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES, 'x_min': x_min, 'x_max':x_max}
SYSTEM_NO = 61
oc.data_gen_sys_arc2s(sys_params, N_CURVES,SYSTEM_NO)
oc.plot_one_curve(SYSTEM_NO,CURVE_NO=0)
oc.plot_training_valid_test_states(SYSTEM_NO)

## EMBEDDED DATA OF THE ABOVE EXAMPLE

# SYSTEM_NO = 62
# EMBEDDING = 4
# SYSTEM_NO = 63
# EMBEDDING = 5
# SYSTEM_NO = 64
# EMBEDDING = 6
SYSTEM_NO = 65
EMBEDDING = 7
storage_folder = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing' + '/System_' + str(SYSTEM_NO)
if os.path.exists(storage_folder):
    shutil.rmtree(storage_folder)
    os.mkdir(storage_folder)
    # get_input = input('Do you wanna delete the existing system[y/n]? ')
    # if get_input == 'y':
    #     shutil.rmtree(storage_folder)
    #     os.mkdir(storage_folder)
    # else:
    #     return
else:
    os.mkdir(storage_folder)
with open('/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_61/System_61_SimulatedData.pickle', 'rb') as handle:
    dict_indexed_data = pickle.load(handle)
N_CURVES = 300
oc.sort_to_DMD_folder(storage_folder, N_CURVES, dict_indexed_data,SYSTEM_NO,EMBEDDING_NUMBER = EMBEDDING )



##
SYSTEM_NO = 28
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_ocDeepDMDdata.pickle', 'rb') as handle:
    d = pickle.load(handle)
print(d.keys())

# ==========================
## System 4 - Combinatorial Promoter system
# ==========================
# System Parameters
k1f = 1e-2; # [1/min] Used kfa
k1r = 0.09; # [1/min]
k2f = 0.005; # [1/min] Used kfr
k2r = 0.2; # [1/min] # Assumed that the activator and the repressor have the same type of dynamics
k3f = 0.55
k3r = 0.1
k4f = 0.184
k4r = 0.002
k5f = 0.092
k5r = 0.25
k6f = 0.82
k6r = 0.15
k7f = 0.85
k7r = 0.1
k8f = 0.1
delta = 0.1
Ka = 0.5
u1 = 5
u2 = 2

x_ACT_min = 1
x_ACT_max = 2

x_REP_min = 0.1
x_REP_max = 2

x_DNA_min = 0.1
x_DNA_max = 2

x_RNAP_min = 0.1
x_RNAP_max = 0.5

x_mRNA_min = 0.1
x_mRNA_max = 0.5

x_min = np.array([x_ACT_min,x_REP_min,x_DNA_min,x_RNAP_min,0,0,0,0,0,0,x_mRNA_min])
x_max = np.array([x_ACT_max,x_REP_max,x_DNA_max,x_RNAP_max,0,0,0,0,0,0,x_mRNA_max])
sys_params_arc4s = (k1f,k1r,k2f,k2r,k3f,k3r,k4f,k4r,k5f,k5r,k6f,k6r,k7f,k7r,k8f,delta,u1,u2)

# Simulation Parameters
sampling_time = 2
simulation_time = 40
N_CURVES = 300
ls_prediction_steps=[1]

sys_params = {'sys_params_arc4s': sys_params_arc4s , 'Ka':Ka, 'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES, 'x_min': x_min, 'x_max':x_max}
SYSTEM_NO = 30
oc.data_gen_sys_combinatorial_promoter(sys_params, N_CURVES,SYSTEM_NO)
oc.plot_training_valid_test_states(SYSTEM_NO)
oc.plot_one_curve(SYSTEM_NO,CURVE_NO=0)


# ==========================
## System 5 - Glycolytic Oscillator
# ==========================
# System Parameters
J0 = 2.5
N = 1.
A = 4.

k1 = 100.
k2 = 6.
k3 = 16.
k4 = 100.
k5 = 1.28
k6 = 12.
k7 = 1.8
K1 = 0.52
kappa = 13.
mu = 0.1
q = 4

Ka = 1
# x_min= np.asarray([0.15,0.19,0.04,0.1,0.08,0.14,0.05]) # original
# x_min= np.asarray([0.15,0.19,0.04,0.1,0.08,0.14,0.05])
# x_max = x_min
x_max = np.asarray([1.6,2.16,0.2,0.35,0.3,2.67,0.1])
x_min = x_max/2
# x_min = 2*x_min
# x_min = x_max
sys_params_arc4s = (k1,k2,k3,k4,k5,k6,k7,K1,kappa,mu,q,J0,N,A)
# Simulation Parameters
sampling_time = 0.02
simulation_time = 5
N_CURVES = 300
ls_prediction_steps=[1]
sys_params = {'sys_params_arc4s': sys_params_arc4s , 'Ka':Ka, 'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES, 'x_min': x_min, 'x_max':x_max}
SYSTEM_NO = 70
oc.data_gen_sys_glycolytic_oscillator(sys_params, N_CURVES,SYSTEM_NO,DOWNSAMPLE_FACTOR = 10)
oc.plot_one_curve(SYSTEM_NO,CURVE_NO=0)
oc.plot_training_valid_test_states(SYSTEM_NO)



# ==========================
## System 6 - Duffing Oscillator
# ==========================
# System Parameters
alpha = 1
beta = -1
delta =0.5
gamma = 0.05

x_min= np.asarray([-2,-2]) # original
x_max= np.asarray([2,2])
sys_params_duffosc = (alpha,beta,delta)
# Simulation Parameters
sampling_time = 0.1
simulation_time = 15
N_CURVES = 300
ls_prediction_steps=[1]
sys_params = {'sys_params_duffosc': sys_params_duffosc , 'gamma': gamma ,'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES, 'x_min': x_min, 'x_max':x_max}
SYSTEM_NO = 80
oc.data_gen_sys_duffing_oscillator(sys_params, N_CURVES,SYSTEM_NO)
oc.plot_one_curve(SYSTEM_NO,CURVE_NO=0)
oc.plot_training_valid_test_states(SYSTEM_NO)

# ==========================
## System 7 - Nonlinear MEMS accelerometer with Capacitive Sensing - MEMS_accel
# ==========================
# System Parameters
m = 1
k1_l = 1
c = 0.4
k3_nl = 2
Vs = 5
d = 3

x_min= np.asarray([-2,-2]) # original
x_max= np.asarray([2,2])
MEMS_accel = (m,k1_l,c,k3_nl)
# Simulation Parameters
sampling_time = 0.5
simulation_time = 15
N_CURVES = 300
ls_prediction_steps=[1]
sys_params = {'sys_params_MEMS_accel': MEMS_accel , 'Vs': Vs, 'd':d ,'Ts': sampling_time, 't_end': simulation_time,'N_CURVES': N_CURVES, 'x_min': x_min, 'x_max':x_max}
SYSTEM_NO = 91
oc.data_gen_sys_MEMS_accelerometer(sys_params, N_CURVES,SYSTEM_NO)
oc.plot_one_curve(SYSTEM_NO,CURVE_NO=0)
oc.plot_training_valid_test_states(SYSTEM_NO)

