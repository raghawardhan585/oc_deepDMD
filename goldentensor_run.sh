#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [process var] [run_no] [n_layers] [n_nodes] [write_to_file] 
python3 hammerstein_nn_identification.py '/gpu:0' 60 'x' 0 6 2 > Run_info/SYS_60_RUN_0.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 60 'x' 1 6 4 > Run_info/SYS_60_RUN_1.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 60 'x' 2 6 8 > Run_info/SYS_60_RUN_2.txt &
python3 hammerstein_nn_identification.py '/gpu:3' 60 'x' 3 6 12 > Run_info/SYS_60_RUN_3.txt &
wait 
python3 hammerstein_nn_identification.py '/gpu:0' 60 'x' 4 8 2 > Run_info/SYS_60_RUN_4.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 60 'x' 5 8 4 > Run_info/SYS_60_RUN_5.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 60 'x' 6 8 8 > Run_info/SYS_60_RUN_6.txt &
python3 hammerstein_nn_identification.py '/gpu:3' 60 'x' 7 8 12 > Run_info/SYS_60_RUN_7.txt &
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
