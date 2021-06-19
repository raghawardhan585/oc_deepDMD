#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 703 0 0 3 10 > Run_info/SYS_703_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 1 0 3 10 > Run_info/SYS_703_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 2 1 3 10 > Run_info/SYS_703_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 3 1 3 10 > Run_info/SYS_703_RUN_3.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 4 2 3 10 > Run_info/SYS_703_RUN_4.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 5 2 3 10 > Run_info/SYS_703_RUN_5.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 6 3 3 10 > Run_info/SYS_703_RUN_6.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 7 3 3 10 > Run_info/SYS_703_RUN_7.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 8 4 3 10 > Run_info/SYS_703_RUN_8.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 9 4 3 10 > Run_info/SYS_703_RUN_9.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 10 5 3 10 > Run_info/SYS_703_RUN_10.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 11 5 3 10 > Run_info/SYS_703_RUN_11.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 12 6 3 10 > Run_info/SYS_703_RUN_12.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 13 6 3 10 > Run_info/SYS_703_RUN_13.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 14 7 3 10 > Run_info/SYS_703_RUN_14.txt &
wait 
python3 deepDMD.py '/cpu:0' 703 15 7 3 10 > Run_info/SYS_703_RUN_15.txt &
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
