#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 61 0 4 10 10 > Run_info/SYS_61_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 61 1 4 10 12 > Run_info/SYS_61_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 61 2 4 9 10 > Run_info/SYS_61_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 61 3 4 9 12 > Run_info/SYS_61_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 61 4 3 9 10 > Run_info/SYS_61_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 61 5 3 9 12 > Run_info/SYS_61_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 61 6 3 10 10 > Run_info/SYS_61_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 61 7 3 10 12 > Run_info/SYS_61_RUN_7.txt &
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
