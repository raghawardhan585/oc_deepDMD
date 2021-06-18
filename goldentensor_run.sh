#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 703 0 0 4 10 > Run_info/SYS_703_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 703 1 0 4 10 > Run_info/SYS_703_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 703 2 0 4 10 > Run_info/SYS_703_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 703 3 1 4 10 > Run_info/SYS_703_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 703 4 1 4 10 > Run_info/SYS_703_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 703 5 1 4 10 > Run_info/SYS_703_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 703 6 2 4 10 > Run_info/SYS_703_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 703 7 2 4 10 > Run_info/SYS_703_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 703 8 2 4 10 > Run_info/SYS_703_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 703 9 3 4 10 > Run_info/SYS_703_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 703 10 3 4 10 > Run_info/SYS_703_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 703 11 3 4 10 > Run_info/SYS_703_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 703 12 4 4 10 > Run_info/SYS_703_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 703 13 4 4 10 > Run_info/SYS_703_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 703 14 4 4 10 > Run_info/SYS_703_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 703 15 5 4 10 > Run_info/SYS_703_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 703 16 5 4 10 > Run_info/SYS_703_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 703 17 5 4 10 > Run_info/SYS_703_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 703 18 6 4 10 > Run_info/SYS_703_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 703 19 6 4 10 > Run_info/SYS_703_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 703 20 6 4 10 > Run_info/SYS_703_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 703 21 7 4 10 > Run_info/SYS_703_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 703 22 7 4 10 > Run_info/SYS_703_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 703 23 7 4 10 > Run_info/SYS_703_RUN_23.txt &
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
