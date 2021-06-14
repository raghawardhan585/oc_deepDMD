#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 602 0 1 3 4 > Run_info/SYS_602_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 602 1 1 3 10 > Run_info/SYS_602_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 602 2 1 4 4 > Run_info/SYS_602_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 602 3 1 4 10 > Run_info/SYS_602_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 602 4 2 3 4 > Run_info/SYS_602_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 602 5 2 3 10 > Run_info/SYS_602_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 602 6 2 4 4 > Run_info/SYS_602_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 602 7 2 4 10 > Run_info/SYS_602_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 602 8 3 3 4 > Run_info/SYS_602_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 602 9 3 3 10 > Run_info/SYS_602_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 602 10 3 4 4 > Run_info/SYS_602_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 602 11 3 4 10 > Run_info/SYS_602_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 602 12 4 3 4 > Run_info/SYS_602_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 602 13 4 3 10 > Run_info/SYS_602_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 602 14 4 4 4 > Run_info/SYS_602_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 602 15 4 4 10 > Run_info/SYS_602_RUN_15.txt &
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
