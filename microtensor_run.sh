#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [process var] [run_no] [n_layers] [n_nodes] [write_to_file] 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 0 3 3 > Run_info/SYS_60_RUN_0.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 1 3 6 > Run_info/SYS_60_RUN_1.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 2 5 3 > Run_info/SYS_60_RUN_2.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 3 5 6 > Run_info/SYS_60_RUN_3.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 4 3 3 > Run_info/SYS_60_RUN_4.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 5 3 6 > Run_info/SYS_60_RUN_5.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 6 5 3 > Run_info/SYS_60_RUN_6.txt &
wait 
python3 hammerstein_nn_identification.py '/cpu:0' 60 'x' 7 5 6 > Run_info/SYS_60_RUN_7.txt &
wait 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
