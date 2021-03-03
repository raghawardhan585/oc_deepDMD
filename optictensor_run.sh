#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 91 0 6 3 15 > Run_info/SYS_91_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 91 1 6 4 6 > Run_info/SYS_91_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 91 2 6 4 12 > Run_info/SYS_91_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 91 3 6 4 15 > Run_info/SYS_91_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 4 7 3 6 > Run_info/SYS_91_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 91 5 7 3 12 > Run_info/SYS_91_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 91 6 7 3 15 > Run_info/SYS_91_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 91 7 7 4 6 > Run_info/SYS_91_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 8 7 9 12 > Run_info/SYS_91_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 91 9 7 9 15 > Run_info/SYS_91_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 91 10 8 3 6 > Run_info/SYS_91_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 91 11 8 3 12 > Run_info/SYS_91_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 12 8 8 15 > Run_info/SYS_91_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 91 13 8 9 6 > Run_info/SYS_91_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 91 14 8 9 12 > Run_info/SYS_91_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 91 15 8 9 15 > Run_info/SYS_91_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 16 9 8 6 > Run_info/SYS_91_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 91 17 9 8 12 > Run_info/SYS_91_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 91 18 9 8 15 > Run_info/SYS_91_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 91 19 9 9 6 > Run_info/SYS_91_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 20 10 4 12 > Run_info/SYS_91_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 91 21 10 4 15 > Run_info/SYS_91_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 91 22 10 8 6 > Run_info/SYS_91_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 91 23 10 8 12 > Run_info/SYS_91_RUN_23.txt &
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
