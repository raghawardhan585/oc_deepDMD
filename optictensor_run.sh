#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [process var] [run_no] [n_layers] [n_nodes] [write_to_file] 
python3 hammerstein_nn_identification.py '/gpu:0' 60 'x' 0 3 9 > Run_info/SYS_60_RUN_0.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 60 'x' 1 3 12 > Run_info/SYS_60_RUN_1.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 60 'x' 2 3 15 > Run_info/SYS_60_RUN_2.txt &
python3 hammerstein_nn_identification.py '/gpu:3' 60 'x' 3 4 3 > Run_info/SYS_60_RUN_3.txt &
wait 
python3 hammerstein_nn_identification.py '/gpu:0' 60 'x' 4 5 9 > Run_info/SYS_60_RUN_4.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 60 'x' 5 5 12 > Run_info/SYS_60_RUN_5.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 60 'x' 6 5 15 > Run_info/SYS_60_RUN_6.txt &
python3 hammerstein_nn_identification.py '/gpu:3' 60 'x' 7 6 3 > Run_info/SYS_60_RUN_7.txt &
wait 
python3 hammerstein_nn_identification.py '/gpu:0' 60 'x' 8 3 9 > Run_info/SYS_60_RUN_8.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 60 'x' 9 3 12 > Run_info/SYS_60_RUN_9.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 60 'x' 10 3 15 > Run_info/SYS_60_RUN_10.txt &
python3 hammerstein_nn_identification.py '/gpu:3' 60 'x' 11 4 3 > Run_info/SYS_60_RUN_11.txt &
wait 
python3 hammerstein_nn_identification.py '/gpu:0' 60 'x' 12 5 9 > Run_info/SYS_60_RUN_12.txt &
python3 hammerstein_nn_identification.py '/gpu:1' 60 'x' 13 5 12 > Run_info/SYS_60_RUN_13.txt &
python3 hammerstein_nn_identification.py '/gpu:2' 60 'x' 14 5 15 > Run_info/SYS_60_RUN_14.txt &
python3 hammerstein_nn_identification.py '/gpu:3' 60 'x' 15 6 3 > Run_info/SYS_60_RUN_15.txt &
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
