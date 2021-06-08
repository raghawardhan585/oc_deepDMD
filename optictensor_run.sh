#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 501 0 4 3 15 > Run_info/SYS_501_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 501 1 4 3 20 > Run_info/SYS_501_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 501 2 4 3 15 > Run_info/SYS_501_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 501 3 4 3 20 > Run_info/SYS_501_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 4 4 4 15 > Run_info/SYS_501_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 501 5 4 4 20 > Run_info/SYS_501_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 501 6 4 4 15 > Run_info/SYS_501_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 501 7 4 4 20 > Run_info/SYS_501_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 8 5 3 15 > Run_info/SYS_501_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 501 9 5 3 20 > Run_info/SYS_501_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 501 10 5 3 15 > Run_info/SYS_501_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 501 11 5 3 20 > Run_info/SYS_501_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 12 5 4 15 > Run_info/SYS_501_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 501 13 5 4 20 > Run_info/SYS_501_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 501 14 5 4 15 > Run_info/SYS_501_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 501 15 5 4 20 > Run_info/SYS_501_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 16 6 3 15 > Run_info/SYS_501_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 501 17 6 3 20 > Run_info/SYS_501_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 501 18 6 3 15 > Run_info/SYS_501_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 501 19 6 3 20 > Run_info/SYS_501_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 20 6 4 15 > Run_info/SYS_501_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 501 21 6 4 20 > Run_info/SYS_501_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 501 22 6 4 15 > Run_info/SYS_501_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 501 23 6 4 20 > Run_info/SYS_501_RUN_23.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 24 7 3 15 > Run_info/SYS_501_RUN_24.txt &
python3 deepDMD.py '/gpu:1' 501 25 7 3 20 > Run_info/SYS_501_RUN_25.txt &
python3 deepDMD.py '/gpu:2' 501 26 7 3 15 > Run_info/SYS_501_RUN_26.txt &
python3 deepDMD.py '/gpu:3' 501 27 7 3 20 > Run_info/SYS_501_RUN_27.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 28 7 4 15 > Run_info/SYS_501_RUN_28.txt &
python3 deepDMD.py '/gpu:1' 501 29 7 4 20 > Run_info/SYS_501_RUN_29.txt &
python3 deepDMD.py '/gpu:2' 501 30 7 4 15 > Run_info/SYS_501_RUN_30.txt &
python3 deepDMD.py '/gpu:3' 501 31 7 4 20 > Run_info/SYS_501_RUN_31.txt &
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
