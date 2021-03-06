##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import itertools
import ocdeepdmd_simulation_examples_helper_functions as oc

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


## Figure 1 - Data of system 1

# System Parameters
A = np.array([[0.86,0.],[0.8,0.4]])
gamma = -0.9
# Simulation Parameters
N_data_points = 30
N_CURVES = 240
sys_params = {'A':A , 'gamma': gamma, 'N_data_points': N_data_points}
# Phase Space Data
dict_data = {}
X0 = np.empty(shape=(0, 2))
i=0
for x1,x2 in itertools.product(list(np.arange(-10,11,2)), list(np.arange(-125,16,20))):
    sys_params['x0'] = np.array([[x1,x2]])
    X0 = np.concatenate([X0, sys_params['x0']], axis=0)
    dict_data[i] = oc.sim_sys_1_2(sys_params)
    i = i+1
# Plot
plt.figure()
alpha = 1.0
epsilon = alpha - 0.01
arrow_length = 0.3
ls_pts = list(range(0,1))
for i in list(dict_data.keys())[0:]:
    for j in ls_pts:
        if np.abs(dict_data[i]['X'][j, 0]) > 1 or j==0:
            plt.plot(dict_data[i]['X'][j, 0], dict_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=5)
    plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.3)
    if np.mod(i,1)==0:
        for j in ls_pts:
            dist = np.sqrt((dict_data[i]['X'][j, 0] - dict_data[i]['X'][j + 1, 0]) ** 2 + (dict_data[i]['X'][j, 1] - dict_data[i]['X'][j + 1, 1]) ** 2)
            x = dict_data[i]['X'][j, 0]
            y = dict_data[i]['X'][j, 1]
            dx = (dict_data[i]['X'][j + 1, 0] - dict_data[i]['X'][j, 0]) * arrow_length
            dy = (dict_data[i]['X'][j + 1, 1] - dict_data[i]['X'][j, 1]) * arrow_length
            # print(x,' ',y,' ',dist)
            if dist<2:
                plt.arrow(x,y,dx,dy,head_width = 0.1,head_length=0.5,alpha=1,color='tab:green')
            else:
                plt.arrow(x, y, dx, dy, head_width=0.3, head_length=3, alpha=1, color='tab:green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot([0],[0],'o',color='tab:red',markersize=10)
plt.xlim([-10,10])
plt.ylim([-126,16])
plt.show()
plt.savefig("Plots/example_1_phase_portrait.svg")
##

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


sys_params_arc2s = (gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m)
Ts = 0.1
t_end = 40
# Simulation Parameters
dict_data = {}
X0 = np.empty(shape=(0, 2))
t = np.arange(0, t_end, Ts)

# Phase Space Data
dict_data = {}
X0 = np.empty(shape=(0, 2))
i=0
for x1,x2 in itertools.product(list(np.arange(1,30.,3.5)), list(np.arange(1,60.,8))):
    dict_data[i]={}
    x0_curr =  np.array([x1,x2])
    X0 = np.concatenate([X0, np.array([[x1,x2]])], axis=0)
    dict_data[i]['X'] = oc.odeint(oc.activator_repressor_clock_2states, x0_curr, t, args=sys_params_arc2s)
    dict_data[i]['Y'] = k_3n * dict_data[i]['X'][:, 1:2] / (k_3d + dict_data[i]['X'][:, 3:4])
    i = i+1

# for items in dict_data.keys():
#     dict_data[items]['X'] = dict_data[items]['X'][:,[1,3]]

# Plot
plt.figure()
alpha = 1.0
epsilon = alpha - 0.01
arrow_length = 1.2
ls_pts = list(range(0,1))
for i in list(dict_data.keys())[0:]:
    for j in ls_pts:
        if np.abs(dict_data[i]['X'][j, 0]) > 1 or j==0:
            plt.plot(dict_data[i]['X'][j, 0], dict_data[i]['X'][j, 1], 'o',color='salmon',fillstyle='none',markersize=5)
    plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.3)
    if np.mod(i,1)==0:
        for j in ls_pts:
            dist = np.sqrt((dict_data[i]['X'][j, 0] - dict_data[i]['X'][j + 1, 0]) ** 2 + (dict_data[i]['X'][j, 1] - dict_data[i]['X'][j + 1, 1]) ** 2)
            x = dict_data[i]['X'][j, 0]
            y = dict_data[i]['X'][j, 1]
            dx = (dict_data[i]['X'][j + 1, 0] - dict_data[i]['X'][j, 0]) * arrow_length
            dy = (dict_data[i]['X'][j + 1, 1] - dict_data[i]['X'][j, 1]) * arrow_length
            # print(x,' ',y,' ',dist)
            if dist<0.1:
                plt.arrow(x,y,dx,dy,head_width = 0.02,head_length=0.03,alpha=1,color='tab:green')
            else:
                plt.arrow(x, y, dx, dy, head_width=1., head_length=0.9, alpha=1, color='tab:green')
plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(dict_data[0]['X'][200:,0],dict_data[0]['X'][200:,1],color='tab:red',markersize=10)
plt.plot(dict_data[5]['X'][200:,0],dict_data[5]['X'][200:,1],color='tab:red',markersize=10)
plt.plot(dict_data[10]['X'][200:,0],dict_data[10]['X'][200:,1],color='tab:red',markersize=10)
# plt.xlim([0,0.6])
# plt.ylim([0,1.65])
plt.xlim([-0.1,30])
plt.ylim([-0.1,60])
plt.show()
plt.savefig("Plots/example_1_phase_portrait.svg")


##

