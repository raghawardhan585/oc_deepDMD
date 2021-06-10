#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 500 0 4 3 15 > Run_info/SYS_500_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 500 1 4 3 20 > Run_info/SYS_500_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 500 2 4 4 15 > Run_info/SYS_500_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 500 3 4 4 20 > Run_info/SYS_500_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 500 4 5 3 15 > Run_info/SYS_500_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 500 5 5 3 20 > Run_info/SYS_500_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 500 6 5 4 15 > Run_info/SYS_500_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 500 7 5 4 20 > Run_info/SYS_500_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 500 8 6 3 15 > Run_info/SYS_500_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 500 9 6 3 20 > Run_info/SYS_500_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 500 10 6 4 15 > Run_info/SYS_500_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 500 11 6 4 20 > Run_info/SYS_500_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 500 12 7 3 15 > Run_info/SYS_500_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 500 13 7 3 20 > Run_info/SYS_500_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 500 14 7 4 15 > Run_info/SYS_500_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 500 15 7 4 20 > Run_info/SYS_500_RUN_15.txt &
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
