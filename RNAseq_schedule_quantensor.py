##
import numpy as np
import Sequential_Helper_Functions as seq
import itertools


## Bash Script Generation
DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR = 402

dict_hp={}
dict_hp['x']={}
dict_hp['x']['ls_dict_size'] = [5]
dict_hp['x']['ls_nn_layers'] = [4]
dict_hp['x']['ls_nn_nodes'] = [15]
dict_hp['y']={}
dict_hp['y']['ls_dict_size'] = [1]
dict_hp['y']['ls_nn_layers'] = [3,4]
dict_hp['y']['ls_nn_nodes'] = [20,25]
dict_hp['xy']={}
dict_hp['xy']['ls_dict_size'] = [2,3,4]
dict_hp['xy']['ls_nn_layers'] = [8,9]
dict_hp['xy']['ls_nn_nodes'] = [6,8]
process_variable = 'x'
SYSTEM_NO = DATA_SYSTEM_TO_WRITE_BASH_SCRIPT_FOR

ls_dict_size = dict_hp[process_variable]['ls_dict_size']
ls_nn_layers = dict_hp[process_variable]['ls_nn_layers']
ls_nn_nodes = dict_hp[process_variable]['ls_nn_nodes']
ls_regularization_parameter = [0.005] #np.arange(4e-6, 4.2e-6, 0.1e-7)#np.concatenate([np.array([0]),np.arange(2e-5, 9.5e-5, 0.5e-5)],axis=0)#np.arange(0, 1e-3, 2.5e-5) #[3.75e-4] #np.arange(5e-5,1e-3,2.5e-5)

# a = list(itertools.product(ls_dict_size,ls_nn_layers,ls_nn_nodes))
a = list(itertools.product(ls_dict_size,ls_nn_layers,ls_nn_nodes,ls_regularization_parameter))
for i in range(len(a)):
    if a[i][0] ==0:
        # a[i] = (0,1,0)
        a[i] = (0, 1, 0, a[i][-1])

print('[INFO] TOTAL NUMBER OF RUNS SCHEDULED : ',len(a))
dict_all_run_conditions ={}
for i in range(len(a)):
    dict_all_run_conditions[i] ={}
    for items in ['x','y','xy']:
        if items != process_variable:
            dict_all_run_conditions[i][items] = {'dict_size': 1, 'nn_layers': 1,'nn_nodes': 1}
        else:
            dict_all_run_conditions[i][process_variable] = {'dict_size': a[i][0], 'nn_layers': a[i][1],'nn_nodes': a[i][2]}
    dict_all_run_conditions[i]['regularization lambda'] = a[i][-1]
print(dict_all_run_conditions)

qt = open('/Users/shara/Desktop/oc_deepDMD/quantensor_run.sh','w')
qt.write('#!/bin/bash \n')

# Scheduling Lasso Regression
# qt.write('python3 RNAseq_LassoModel_X.py > System_401/Lasso_Regression_X/LassoRun.txt &\n')
# qt.write('wait \n')
# qt.write('echo "All sessions are complete" \n')
# qt.write('echo "=======================================================" \n')
# qt.close()
# Scheduling oc deepDMD runs
qt.write('rm -rf _current_run_saved_files \n')
qt.write('mkdir _current_run_saved_files \n')
qt.write('rm -rf Run_info \n')
qt.write('mkdir Run_info \n')
qt.write('# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] \n')
run_no = 0
for i in dict_all_run_conditions.keys():
    run_params = ''
    for items in ['x','y','xy']:
        for sub_items in dict_all_run_conditions[i][items].keys():
            run_params = run_params + ' ' + str(dict_all_run_conditions[i][items][sub_items])
    run_params = run_params + ' ' + str(dict_all_run_conditions[i]['regularization lambda'])
    general_run = 'python3 ocdeepDMD_Sequential.py \'/cpu:0\' ' + str(SYSTEM_NO) + ' ' + str(run_no) + ' '
    write_to_file = ' > Run_info/SYS_' + str(SYSTEM_NO) + '_RUN_' + str(run_no) + '.txt &\n'
    qt.write(general_run + run_params + write_to_file)
    qt.write('wait \n')
    run_no = run_no + 1

qt.write('wait \n')
qt.write('echo "All sessions are complete" \n')
qt.write('echo "=======================================================" \n')
qt.write('cd .. \n')
qt.write('rm -R _current_run_saved_files \n')
qt.write('rm -R Run_info \n')
qt.write('cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files \n')
qt.write('cp -a oc_deepDMD/Run_info/ Run_info \n')
qt.write('cd oc_deepDMD/ \n')
qt.close()


## Transfer the oc deepDMD files
seq.transfer_current_ocDeepDMD_run_files()


##

