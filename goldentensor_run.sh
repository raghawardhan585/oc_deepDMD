#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 800 0 0 4 10 > Run_info/SYS_800_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 800 1 0 4 10 > Run_info/SYS_800_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 800 2 0 4 10 > Run_info/SYS_800_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 800 3 1 4 10 > Run_info/SYS_800_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 800 4 1 4 10 > Run_info/SYS_800_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 800 5 1 4 10 > Run_info/SYS_800_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 800 6 2 4 10 > Run_info/SYS_800_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 800 7 2 4 10 > Run_info/SYS_800_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 800 8 2 4 10 > Run_info/SYS_800_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 800 9 3 4 10 > Run_info/SYS_800_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 800 10 3 4 10 > Run_info/SYS_800_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 800 11 3 4 10 > Run_info/SYS_800_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 800 12 4 4 10 > Run_info/SYS_800_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 800 13 4 4 10 > Run_info/SYS_800_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 800 14 4 4 10 > Run_info/SYS_800_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 800 15 5 4 10 > Run_info/SYS_800_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 800 16 5 4 10 > Run_info/SYS_800_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 800 17 5 4 10 > Run_info/SYS_800_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 800 18 6 4 10 > Run_info/SYS_800_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 800 19 6 4 10 > Run_info/SYS_800_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 800 20 6 4 10 > Run_info/SYS_800_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 800 21 7 4 10 > Run_info/SYS_800_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 800 22 7 4 10 > Run_info/SYS_800_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 800 23 7 4 10 > Run_info/SYS_800_RUN_23.txt &
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
