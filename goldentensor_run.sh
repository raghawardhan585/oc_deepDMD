#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 601 0 0 4 7 > Run_info/SYS_601_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 601 1 0 4 10 > Run_info/SYS_601_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 601 2 0 5 7 > Run_info/SYS_601_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 601 3 0 5 10 > Run_info/SYS_601_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 4 1 4 7 > Run_info/SYS_601_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 601 5 1 4 10 > Run_info/SYS_601_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 601 6 1 5 7 > Run_info/SYS_601_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 601 7 1 5 10 > Run_info/SYS_601_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 8 2 4 7 > Run_info/SYS_601_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 601 9 2 4 10 > Run_info/SYS_601_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 601 10 2 5 7 > Run_info/SYS_601_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 601 11 2 5 10 > Run_info/SYS_601_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 12 3 4 7 > Run_info/SYS_601_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 601 13 3 4 10 > Run_info/SYS_601_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 601 14 3 5 7 > Run_info/SYS_601_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 601 15 3 5 10 > Run_info/SYS_601_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 16 4 4 7 > Run_info/SYS_601_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 601 17 4 4 10 > Run_info/SYS_601_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 601 18 4 5 7 > Run_info/SYS_601_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 601 19 4 5 10 > Run_info/SYS_601_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 20 5 4 7 > Run_info/SYS_601_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 601 21 5 4 10 > Run_info/SYS_601_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 601 22 5 5 7 > Run_info/SYS_601_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 601 23 5 5 10 > Run_info/SYS_601_RUN_23.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 24 6 4 7 > Run_info/SYS_601_RUN_24.txt &
python3 deepDMD.py '/gpu:1' 601 25 6 4 10 > Run_info/SYS_601_RUN_25.txt &
python3 deepDMD.py '/gpu:2' 601 26 6 5 7 > Run_info/SYS_601_RUN_26.txt &
python3 deepDMD.py '/gpu:3' 601 27 6 5 10 > Run_info/SYS_601_RUN_27.txt &
wait 
python3 deepDMD.py '/gpu:0' 601 28 7 4 7 > Run_info/SYS_601_RUN_28.txt &
python3 deepDMD.py '/gpu:1' 601 29 7 4 10 > Run_info/SYS_601_RUN_29.txt &
python3 deepDMD.py '/gpu:2' 601 30 7 5 7 > Run_info/SYS_601_RUN_30.txt &
python3 deepDMD.py '/gpu:3' 601 31 7 5 10 > Run_info/SYS_601_RUN_31.txt &
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
