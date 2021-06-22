#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 706 0 0 4 10 > Run_info/SYS_706_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 706 1 0 4 10 > Run_info/SYS_706_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 706 2 1 4 10 > Run_info/SYS_706_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 706 3 1 4 10 > Run_info/SYS_706_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 706 4 2 4 10 > Run_info/SYS_706_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 706 5 2 4 10 > Run_info/SYS_706_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 706 6 3 4 10 > Run_info/SYS_706_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 706 7 3 4 10 > Run_info/SYS_706_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 706 8 4 4 10 > Run_info/SYS_706_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 706 9 4 4 10 > Run_info/SYS_706_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 706 10 5 4 10 > Run_info/SYS_706_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 706 11 5 4 10 > Run_info/SYS_706_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 706 12 6 4 10 > Run_info/SYS_706_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 706 13 6 4 10 > Run_info/SYS_706_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 706 14 7 4 10 > Run_info/SYS_706_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 706 15 7 4 10 > Run_info/SYS_706_RUN_15.txt &
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
