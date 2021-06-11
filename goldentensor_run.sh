#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 411 0 0 4 10 > Run_info/SYS_411_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 411 1 0 4 10 > Run_info/SYS_411_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 411 2 0 4 10 > Run_info/SYS_411_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 411 3 1 4 10 > Run_info/SYS_411_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 411 4 1 4 10 > Run_info/SYS_411_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 411 5 1 4 10 > Run_info/SYS_411_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 411 6 2 4 10 > Run_info/SYS_411_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 411 7 2 4 10 > Run_info/SYS_411_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 411 8 2 4 10 > Run_info/SYS_411_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 411 9 3 4 10 > Run_info/SYS_411_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 411 10 3 4 10 > Run_info/SYS_411_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 411 11 3 4 10 > Run_info/SYS_411_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 411 12 4 4 10 > Run_info/SYS_411_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 411 13 4 4 10 > Run_info/SYS_411_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 411 14 4 4 10 > Run_info/SYS_411_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 411 15 5 4 10 > Run_info/SYS_411_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 411 16 5 4 10 > Run_info/SYS_411_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 411 17 5 4 10 > Run_info/SYS_411_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 411 18 6 4 10 > Run_info/SYS_411_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 411 19 6 4 10 > Run_info/SYS_411_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 411 20 6 4 10 > Run_info/SYS_411_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 411 21 7 4 10 > Run_info/SYS_411_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 411 22 7 4 10 > Run_info/SYS_411_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 411 23 7 4 10 > Run_info/SYS_411_RUN_23.txt &
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
