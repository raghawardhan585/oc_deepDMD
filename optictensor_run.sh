#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 80 0 3 3 12 > Run_info/SYS_80_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 80 1 3 4 6 > Run_info/SYS_80_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 80 2 3 4 9 > Run_info/SYS_80_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 80 3 3 4 12 > Run_info/SYS_80_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 4 3 7 6 > Run_info/SYS_80_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 80 5 3 7 9 > Run_info/SYS_80_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 80 6 3 7 12 > Run_info/SYS_80_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 80 7 4 3 6 > Run_info/SYS_80_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 8 4 5 9 > Run_info/SYS_80_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 80 9 4 5 12 > Run_info/SYS_80_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 80 10 4 6 6 > Run_info/SYS_80_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 80 11 4 6 9 > Run_info/SYS_80_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 12 5 3 12 > Run_info/SYS_80_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 80 13 5 4 6 > Run_info/SYS_80_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 80 14 5 4 9 > Run_info/SYS_80_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 80 15 5 4 12 > Run_info/SYS_80_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 16 5 7 6 > Run_info/SYS_80_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 80 17 5 7 9 > Run_info/SYS_80_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 80 18 5 7 12 > Run_info/SYS_80_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 80 19 6 3 6 > Run_info/SYS_80_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 80 20 6 5 9 > Run_info/SYS_80_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 80 21 6 5 12 > Run_info/SYS_80_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 80 22 6 6 6 > Run_info/SYS_80_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 80 23 6 6 9 > Run_info/SYS_80_RUN_23.txt &
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
