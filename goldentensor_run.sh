#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 11 0 3 7 5 > Run_info/SYS_11_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 11 1 3 7 6 > Run_info/SYS_11_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 11 2 3 8 5 > Run_info/SYS_11_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 11 3 3 8 6 > Run_info/SYS_11_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 11 4 3 9 5 > Run_info/SYS_11_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 11 5 3 9 6 > Run_info/SYS_11_RUN_5.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
