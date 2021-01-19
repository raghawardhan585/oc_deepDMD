#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 10 0 3 5 15 > Run_info/SYS_10_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 10 1 3 5 18 > Run_info/SYS_10_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 10 2 3 6 9 > Run_info/SYS_10_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 10 3 3 6 12 > Run_info/SYS_10_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 10 4 6 4 9 > Run_info/SYS_10_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 10 5 6 4 12 > Run_info/SYS_10_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 10 6 6 4 15 > Run_info/SYS_10_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 10 7 6 4 18 > Run_info/SYS_10_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 10 8 6 6 15 > Run_info/SYS_10_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 10 9 6 6 18 > Run_info/SYS_10_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 10 10 6 7 9 > Run_info/SYS_10_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 10 11 6 7 12 > Run_info/SYS_10_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 10 12 9 5 9 > Run_info/SYS_10_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 10 13 9 5 12 > Run_info/SYS_10_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 10 14 9 5 15 > Run_info/SYS_10_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 10 15 9 5 18 > Run_info/SYS_10_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 10 16 9 7 15 > Run_info/SYS_10_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 10 17 9 7 18 > Run_info/SYS_10_RUN_17.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
