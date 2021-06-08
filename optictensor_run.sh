#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 502 0 4 3 20 > Run_info/SYS_502_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 502 1 4 4 20 > Run_info/SYS_502_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 502 2 5 3 20 > Run_info/SYS_502_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 502 3 5 4 20 > Run_info/SYS_502_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 502 4 6 3 20 > Run_info/SYS_502_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 502 5 6 4 20 > Run_info/SYS_502_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 502 6 7 3 20 > Run_info/SYS_502_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 502 7 7 4 20 > Run_info/SYS_502_RUN_7.txt &
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
