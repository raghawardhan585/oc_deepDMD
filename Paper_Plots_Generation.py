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
    plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.5)
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

##

# System Parameters
mA_init = 1
mB_init = 1
gamma_A = 0.7
gamma_B = 0.5
delta_A = 1.
delta_B = 1.
alpha_A0= 0.4
alpha_B0= 0.004
alpha_A = 2.
alpha_B = 2.
K_A = 0.1
K_B = 0.08
kappa_A = 0.9
kappa_B = 0.5
n = 2.
m = 4.
sys_params_arc4s = (gamma_A,gamma_B,delta_A,delta_B,alpha_A0,alpha_B0,alpha_A,alpha_B,K_A,K_B,kappa_A,kappa_B,n,m)
# k_3n = 3.
# k_3d = 1.08
Ts = 0.5
t_end = 30

# Simulation Parameters
dict_data = {}
X0 = np.empty(shape=(0, 2))
t = np.arange(0, t_end, Ts)




for i in range(N_CURVES):
    x0_curr = np.random.uniform(sys_params['x_min'], sys_params['x_max'], size=(4))
    X0 = np.concatenate([X0, x0_curr.reshape(1, -1)], axis=0)
    X = odeint(activator_repressor_clock_4states, x0_curr, t, args=sys_params['sys_params_arc4s'])
    Y = sys_params['k_3n'] * X[:, 1:2] / (sys_params['k_3d'] + X[:, 3:4])
    dict_indexed_data[i] = {'X': X, 'Y': Y}
    plt.plot(dict_indexed_data[i]['X'][:, 1], dict_indexed_data[i]['X'][:, 3])



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
    plt.plot(dict_data[i]['X'][:, 0], dict_data[i]['X'][:, 1], color='tab:blue',linewidth=0.5)
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