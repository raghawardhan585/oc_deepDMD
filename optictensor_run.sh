#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 502 0 10 3 20 > Run_info/SYS_502_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 502 1 10 4 20 > Run_info/SYS_502_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 502 2 11 3 20 > Run_info/SYS_502_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 502 3 11 4 20 > Run_info/SYS_502_RUN_3.txt &
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
