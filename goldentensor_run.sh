#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [process var] [run_no] [n_layers] [n_nodes] [write_to_file] 
python3 hammerstein_nn_identification.py '/gpu:0' 10 'x' 0 5 5 > Run_info/SYS_10_RUN_0.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 10 'x' 1 5 10 > Run_info/SYS_10_RUN_1.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 10 'x' 2 5 15 > Run_info/SYS_10_RUN_2.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
