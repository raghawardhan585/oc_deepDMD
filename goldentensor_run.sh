#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 60 0 5 3 10 > Run_info/SYS_60_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 60 1 5 3 15 > Run_info/SYS_60_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 60 2 5 3 20 > Run_info/SYS_60_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 60 3 5 4 10 > Run_info/SYS_60_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 60 4 8 4 15 > Run_info/SYS_60_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 60 5 8 4 20 > Run_info/SYS_60_RUN_5.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
