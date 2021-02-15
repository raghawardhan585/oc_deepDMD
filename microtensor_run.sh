#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 80 0 3 3 6 > Run_info/SYS_80_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 1 3 3 9 > Run_info/SYS_80_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 2 3 6 9 > Run_info/SYS_80_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 3 3 6 12 > Run_info/SYS_80_RUN_3.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 4 4 4 12 > Run_info/SYS_80_RUN_4.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 5 4 5 6 > Run_info/SYS_80_RUN_5.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 6 5 3 6 > Run_info/SYS_80_RUN_6.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 7 5 3 9 > Run_info/SYS_80_RUN_7.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 8 5 6 9 > Run_info/SYS_80_RUN_8.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 9 5 6 12 > Run_info/SYS_80_RUN_9.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 10 6 4 12 > Run_info/SYS_80_RUN_10.txt &
wait 
python3 deepDMD.py '/cpu:0' 80 11 6 5 6 > Run_info/SYS_80_RUN_11.txt &
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
