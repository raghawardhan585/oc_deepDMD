#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 501 0 0 3 6 > Run_info/SYS_501_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 501 1 0 3 10 > Run_info/SYS_501_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 501 2 0 3 6 > Run_info/SYS_501_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 501 3 0 3 10 > Run_info/SYS_501_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 4 0 4 6 > Run_info/SYS_501_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 501 5 0 4 10 > Run_info/SYS_501_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 501 6 0 4 6 > Run_info/SYS_501_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 501 7 0 4 10 > Run_info/SYS_501_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 8 1 3 6 > Run_info/SYS_501_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 501 9 1 3 10 > Run_info/SYS_501_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 501 10 1 3 6 > Run_info/SYS_501_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 501 11 1 3 10 > Run_info/SYS_501_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 12 1 4 6 > Run_info/SYS_501_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 501 13 1 4 10 > Run_info/SYS_501_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 501 14 1 4 6 > Run_info/SYS_501_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 501 15 1 4 10 > Run_info/SYS_501_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 16 2 3 6 > Run_info/SYS_501_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 501 17 2 3 10 > Run_info/SYS_501_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 501 18 2 3 6 > Run_info/SYS_501_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 501 19 2 3 10 > Run_info/SYS_501_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 20 2 4 6 > Run_info/SYS_501_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 501 21 2 4 10 > Run_info/SYS_501_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 501 22 2 4 6 > Run_info/SYS_501_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 501 23 2 4 10 > Run_info/SYS_501_RUN_23.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 24 3 3 6 > Run_info/SYS_501_RUN_24.txt &
python3 deepDMD.py '/gpu:1' 501 25 3 3 10 > Run_info/SYS_501_RUN_25.txt &
python3 deepDMD.py '/gpu:2' 501 26 3 3 6 > Run_info/SYS_501_RUN_26.txt &
python3 deepDMD.py '/gpu:3' 501 27 3 3 10 > Run_info/SYS_501_RUN_27.txt &
wait 
python3 deepDMD.py '/gpu:0' 501 28 3 4 6 > Run_info/SYS_501_RUN_28.txt &
python3 deepDMD.py '/gpu:1' 501 29 3 4 10 > Run_info/SYS_501_RUN_29.txt &
python3 deepDMD.py '/gpu:2' 501 30 3 4 6 > Run_info/SYS_501_RUN_30.txt &
python3 deepDMD.py '/gpu:3' 501 31 3 4 10 > Run_info/SYS_501_RUN_31.txt &
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
