#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 10 0 3 4 5 > Run_info/SYS_10_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 1 3 4 6 > Run_info/SYS_10_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 2 3 6 11 > Run_info/SYS_10_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 3 3 6 12 > Run_info/SYS_10_RUN_3.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 4 4 5 5 > Run_info/SYS_10_RUN_4.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 5 4 5 6 > Run_info/SYS_10_RUN_5.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 6 4 7 11 > Run_info/SYS_10_RUN_6.txt &
wait 
python3 deepDMD.py '/cpu:0' 10 7 4 7 12 > Run_info/SYS_10_RUN_7.txt &
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
