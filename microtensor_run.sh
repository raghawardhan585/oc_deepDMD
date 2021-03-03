#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 91 0 6 3 6 > Run_info/SYS_91_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 1 6 3 12 > Run_info/SYS_91_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 2 6 9 12 > Run_info/SYS_91_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 3 6 9 15 > Run_info/SYS_91_RUN_3.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 4 7 8 15 > Run_info/SYS_91_RUN_4.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 5 7 9 6 > Run_info/SYS_91_RUN_5.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 6 8 8 6 > Run_info/SYS_91_RUN_6.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 7 8 8 12 > Run_info/SYS_91_RUN_7.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 8 9 4 12 > Run_info/SYS_91_RUN_8.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 9 9 4 15 > Run_info/SYS_91_RUN_9.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 10 10 3 15 > Run_info/SYS_91_RUN_10.txt &
wait 
python3 deepDMD.py '/cpu:0' 91 11 10 4 6 > Run_info/SYS_91_RUN_11.txt &
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
