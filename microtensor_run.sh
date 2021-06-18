#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 701 0 0 4 10 > Run_info/SYS_701_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 1 0 4 10 > Run_info/SYS_701_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 2 1 4 10 > Run_info/SYS_701_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 3 1 4 10 > Run_info/SYS_701_RUN_3.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 4 2 4 10 > Run_info/SYS_701_RUN_4.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 5 2 4 10 > Run_info/SYS_701_RUN_5.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 6 3 4 10 > Run_info/SYS_701_RUN_6.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 7 3 4 10 > Run_info/SYS_701_RUN_7.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 8 4 4 10 > Run_info/SYS_701_RUN_8.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 9 4 4 10 > Run_info/SYS_701_RUN_9.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 10 5 4 10 > Run_info/SYS_701_RUN_10.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 11 5 4 10 > Run_info/SYS_701_RUN_11.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 12 6 4 10 > Run_info/SYS_701_RUN_12.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 13 6 4 10 > Run_info/SYS_701_RUN_13.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 14 7 4 10 > Run_info/SYS_701_RUN_14.txt &
wait 
python3 deepDMD.py '/cpu:0' 701 15 7 4 10 > Run_info/SYS_701_RUN_15.txt &
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
