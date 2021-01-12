#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 60 0 3 4 7 > Run_info/SYS_60_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 60 1 3 5 3 > Run_info/SYS_60_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 60 2 3 5 5 > Run_info/SYS_60_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 60 3 3 5 7 > Run_info/SYS_60_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 60 4 4 5 3 > Run_info/SYS_60_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 60 5 4 5 5 > Run_info/SYS_60_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 60 6 4 5 7 > Run_info/SYS_60_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 60 7 4 6 3 > Run_info/SYS_60_RUN_7.txt &
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
