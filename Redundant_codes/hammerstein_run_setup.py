##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import random
from Redundant_codes import hammerstein_helper_functions as hm
import itertools

colors = [[0.68627453, 0.12156863, 0.16470589],
          [0.96862745, 0.84705883, 0.40000001],
          [0.83137256, 0.53333336, 0.6156863],
          [0.03529412, 0.01960784, 0.14509805],
          [0.90980393, 0.59607846, 0.78039217],
          [0.69803923, 0.87843138, 0.72941178],
          [0.20784314, 0.81568629, 0.89411765]];
colors = np.asarray(colors);  # defines a color palette


## Bash Script Generation
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 60
NO_OF_ITERATIONS = 1

dict_hp={}
dict_hp['x']={}
dict_hp['x']['ls_nn_layers'] = [5,6,7,8]
dict_hp['x']['ls_nn_nodes'] = [1,2,4,8,12]
# dict_hp['x']['ls_nn_layers'] = [7,8,9]
# dict_hp['x']['ls_nn_nodes'] = [3,6,9]
dict_hp['y']={}
dict_hp['y']['ls_nn_layers'] = [4,5,6]
dict_hp['y']['ls_nn_nodes'] = [1,2,4,8]
process_variable = 'y'
SYSTEM_NO = DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR

ls_nn_layers = dict_hp[process_variable]['ls_nn_layers']
ls_nn_nodes = dict_hp[process_variable]['ls_nn_nodes']
a_base = list(itertools.product(ls_nn_layers,ls_nn_nodes))
a = []
for j in range(NO_OF_ITERATIONS):
    [a.append(items) for items in a_base]
print('[INFO] TOTAL NUMBER OF RUNS SCHEDULED : ',len(a))
dict_all_run_conditions ={}
for i in range(len(a)):
        dict_all_run_conditions[i] = str(a[i][0]) + ' '  + str(a[i][1])
print(dict_all_run_conditions)
# Scheduling
mt = open('/Users/shara/Desktop/oc_deepDMD/microtensor_run.sh','w')
gt = open('/Users/shara/Desktop/oc_deepDMD/goldentensor_run.sh','w')
ot = open('/Users/shara/Desktop/oc_deepDMD/optictensor_run.sh','w')
ls_files  = [mt,gt,ot]
for items in ls_files:
    items.write('#!/bin/bash \n')
    items.write('rm -rf _current_run_saved_files \n')
    items.write('mkdir _current_run_saved_files \n')
    items.write('rm -rf Run_info \n')
    items.write('mkdir Run_info \n')
    items.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [process var] [run_no] [n_layers] [n_nodes] [write_to_file] \n')
running_code = 'hammerstein_nn_identification.py'
ls_run_no =[0,0,0]
for i in dict_all_run_conditions.keys():
    if np.mod(i,10) ==0 or np.mod(i,10) ==1: # Microtensor CPU 0
        general_run = 'python3 ' + running_code + ' \'/cpu:0\' ' + str(SYSTEM_NO) + ' \'' + process_variable + '\' ' + str(ls_run_no[0]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[0]) + '.txt &\n'
        ls_files[0].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_files[0].write('wait \n')
        ls_run_no[0] = ls_run_no[0] + 1
    elif np.mod(i,10)==9: # Goldentensor GPU 3
        general_run = 'python3 ' + running_code + ' \'/gpu:3\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_files[1].write('wait \n')
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 8: # Goldentensor GPU 2
        general_run = 'python3 ' + running_code + ' \'/gpu:2\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' ' + str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 7: # Goldentensor GPU 1
        general_run = 'python3 ' + running_code + ' \'/gpu:1\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10)==6: # Goldentensor GPU 0
        general_run = 'python3 ' + running_code + ' \'/gpu:0\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[1]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[1]) + '.txt &\n'
        ls_files[1].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[1] = ls_run_no[1] + 1
    elif np.mod(i,10) == 5: # Optictensor GPU 3
        general_run = 'python3 ' + running_code + ' \'/gpu:3\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_files[2].write('wait \n')
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10) == 4: # Optictensor GPU 2
        general_run = 'python3 ' + running_code + ' \'/gpu:2\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10)==3: # Optictensor GPU 1
        general_run = 'python3 ' + running_code + ' \'/gpu:1\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
    elif np.mod(i,10) == 2: # Optictensor GPU 0
        general_run = 'python3 ' + running_code + ' \'/gpu:0\' ' + str(SYSTEM_NO) + ' \''+ process_variable + '\' '+ str(ls_run_no[2]) + ' '
        write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(ls_run_no[2]) + '.txt &\n'
        ls_files[2].write(general_run + dict_all_run_conditions[i] + write_to_file)
        ls_run_no[2] = ls_run_no[2] + 1
for items in ls_files:
    items.write('wait \n')
    items.write('echo "All sessions are complete" \n')
    items.write('echo "=======================================================" \n')
    items.write('cd .. \n')
    items.write('rm -R _current_run_saved_files \n')
    items.write('rm -R Run_info \n')
    items.write('cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files \n')
    items.write('cp -a oc_deepDMD/Run_info/ Run_info \n')
    items.write('cd oc_deepDMD/ \n')
    items.close()

##
hm.transfer_current_ocDeepDMD_run_files()
## RUN 1 PROCESSING - Generate predictions and error
# SYSTEM_NO = 10
# ls_process_runs = list(range(0,12))
# OPT_X_RUN = 10
# ls_process_runs = list(range(12,24))


SYSTEM_NO = 60
# ls_process_runs = list(range(0,20))
OPT_X_RUN = 7
ls_process_runs = list(range(20,32))

# SYSTEM_NO = 53
# ls_process_runs = list(range(0,20))
# OPT_X_RUN = 8
# ls_process_runs = list(range(20,29))
# ls_process_runs = list(range(0,29))


hm.generate_predictions_pickle_file(SYSTEM_NO,ls_process_runs,OPT_X_RUN)
sys_folder_name = '/Users/shara/Box/YeungLabUCSBShare/Shara/DoE_Pputida_RNASeq_DataProcessing/System_' + str(SYSTEM_NO)
with open(sys_folder_name + '/dict_predictions_HAMMERSTEIN.pickle','rb') as handle:
    d = pickle.load(handle)
N_CURVES = len(d[list(d.keys())[0]].keys())
ls_train_curves = list(range(int(np.floor(N_CURVES / 3))))
ls_valid_curves = list(range(ls_train_curves[-1] + 1, ls_train_curves[-1] + 1 + int(np.floor(N_CURVES / 3))))
ls_test_curves = list(range(ls_valid_curves[-1] + 1, N_CURVES))
ls_all_available_runs = list(d.keys())
ls_runs = set(ls_process_runs).intersection(set(ls_all_available_runs))
dict_error = {}
for run_no in ls_runs:
    print(run_no)
    dict_error[run_no] = {}
    dict_error[run_no]['train'] = hm.get_error(ls_train_curves, d[run_no])
    dict_error[run_no]['valid'] = hm.get_error(ls_valid_curves, d[run_no])
    dict_error[run_no]['test'] = hm.get_error(ls_test_curves, d[run_no])
df_error_HAMMERSTEIN = pd.DataFrame(dict_error).T
df_err_opt = pd.DataFrame(df_error_HAMMERSTEIN.train + df_error_HAMMERSTEIN.valid)
opt_run = df_err_opt[df_err_opt == df_err_opt.min()].first_valid_index()
print('Optimal Run: ', opt_run)
##
dopt = d[opt_run]
def plot_fit_XY(dict_run,plot_params,ls_runs,scaled=False,one_step = False):
    n_rows = 3
    n_cols = 3
    graphs_per_run = 2
    f,ax = plt.subplots(n_rows,n_cols,sharex=True,figsize = (plot_params['individual_fig_width']*n_cols,plot_params['individual_fig_height']*n_rows))
    i = 0
    for row_i in range(n_rows):
        for col_i in list(range(0,n_cols)):
            if scaled:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X_scaled'].shape[1]
                for j in range(n_states):
                    ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_scaled'][:, j], '.', color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1))
                    if one_step:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_one_step_scaled'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),
                                              label='x' + str(j + 1) + '[scaled]')
                    else:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_n_step_scaled'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),
                                          label='x' + str(j + 1)+ '[scaled]')
                ax[row_i, col_i].legend()
                try:
                    for j in range(dict_run[ls_runs[i]]['Y_scaled'].shape[1]):
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_scaled'][:, j], '.', color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1))
                        if one_step:
                            ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_one_step_scaled'][:, j],color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1), label='y' + str(j + 1) + '[scaled]')
                        else:
                            ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_n_step_scaled'][:, j], color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1), label='y' + str(j + 1)+ '[scaled]')
                    ax[row_i, col_i].legend()
                except:
                    print('No output to plot')
            else:
                # Plot states and outputs
                n_states = dict_run[ls_runs[i]]['X'].shape[1]
                for j in range(n_states):
                    ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X'][:,j],'.',color = colors[np.mod(j,7)], linewidth=int(j / 7 + 1))
                    if one_step:
                        ax[row_i, col_i].plot(dict_run[ls_runs[i]]['X_one_step'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),
                                              label='x' + str(j + 1))
                    else:
                        ax[row_i,col_i].plot(dict_run[ls_runs[i]]['X_n_step'][:, j], color=colors[np.mod(j,7)], linewidth=int(j / 7 + 1),label ='x'+str(j+1) )
                try:
                    for j in range(dict_run[ls_runs[i]]['Y'].shape[1]):
                        ax[row_i,col_i].plot(dict_run[ls_runs[i]]['Y'][:,j],'.',color = colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1))
                        if one_step:
                            ax[row_i, col_i].plot(dict_run[ls_runs[i]]['Y_one_step'][:, j],
                                                      color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1), label='y' + str(j + 1))
                        else:
                            ax[row_i,col_i].plot(dict_run[ls_runs[i]]['Y_n_step'][:, j], color=colors[np.mod(n_states + j,7)], linewidth=int((n_states + j )/ 7 + 1),label ='y'+str(j+1))
                    ax[row_i, col_i].legend()
                except:
                    print('No output to plot')
            i = i+1
            if i == len(ls_runs):
                f.show()
                return f
    f.show()
    return f

with open(sys_folder_name + '/System_' + str(SYSTEM_NO) + '_SimulatedData.pickle','rb') as handle:
    var_i = pickle.load(handle)
N_CURVES = len(var_i.keys())
del var_i
plot_params ={}
plot_params['individual_fig_height'] = 4 #2
plot_params['individual_fig_width'] = 4#2.4
ls_train_curves = list(range(int(np.floor(N_CURVES/3))))
ls_valid_curves = list(range(ls_train_curves[-1] + 1 ,ls_train_curves[-1] + 1 + int(np.floor(N_CURVES/3))))
ls_test_curves = list(range(ls_valid_curves[-1]+1,N_CURVES))
random.shuffle(ls_train_curves)
random.shuffle(ls_valid_curves)
random.shuffle(ls_test_curves)
# f1 = plot_fit_XY(d[opt_run],plot_params,ls_train_curves[0:20],scaled=True,one_step=True)
# f1 = plot_fit_XY(d[opt_run],plot_params,ls_train_curves[0:20],scaled=False,one_step=False)
# f1 = plot_fit_XY(d[opt_run],plot_params,ls_test_curves[0:20],scaled=False,one_step=False)
f1 = plot_fit_XY({k:dopt[k] for k in ls_test_curves[0:9]} ,plot_params,ls_test_curves[0:9],scaled= False,one_step=False)


for i in range(200,300):#d[28].keys():
    SSE = np.sum(np.square(d[28][i]['X'] - d[28][i]['X_n_step'])) +  np.sum(np.square(d[28][i]['Y'] - d[28][i]['Y_one_step']))
    SST = np.sum(np.square(d[28][i]['X'])) + np.sum(np.square(d[28][i]['Y']))
    r2 = np.max([0,(1-SSE/SST)])*100
    print(r2)
    if r2 <98:
        print('Curve: ', i, 'r2: ', r2)
##
# m = 0
# plt.plot(di['X'][:,m],label ='True')
# plt.plot(di['X_n_step'][:,m],label = 'n - step')
# plt.plot(di['X_one_step'][:,m], label = '1 - step')
# plt.legend()
# plt.show()