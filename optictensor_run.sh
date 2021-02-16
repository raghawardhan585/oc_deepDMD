#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 80 0 7 8 12 > Run_info/SYS_80_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 80 1 7 9 6 > Run_info/SYS_80_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 80 2 7 9 9 > Run_info/SYS_80_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 80 3 7 9 12 > Run_info/SYS_80_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 4 8 9 6 > Run_info/SYS_80_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 80 5 8 9 9 > Run_info/SYS_80_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 80 6 8 9 12 > Run_info/SYS_80_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 80 7 8 10 6 > Run_info/SYS_80_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 8 9 9 9 > Run_info/SYS_80_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 80 9 9 9 12 > Run_info/SYS_80_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 80 10 9 10 6 > Run_info/SYS_80_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 80 11 9 10 9 > Run_info/SYS_80_RUN_11.txt &
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
