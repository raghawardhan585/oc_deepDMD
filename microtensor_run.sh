#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 61 0 4 9 10 > Run_info/SYS_61_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 61 1 4 9 12 > Run_info/SYS_61_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 61 2 3 10 10 > Run_info/SYS_61_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 61 3 3 10 12 > Run_info/SYS_61_RUN_3.txt &
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
