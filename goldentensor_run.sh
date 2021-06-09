#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 500 0 0 3 5 > Run_info/SYS_500_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 500 1 0 3 10 > Run_info/SYS_500_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 500 2 0 4 5 > Run_info/SYS_500_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 500 3 0 4 10 > Run_info/SYS_500_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 500 4 1 3 5 > Run_info/SYS_500_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 500 5 1 3 10 > Run_info/SYS_500_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 500 6 1 4 5 > Run_info/SYS_500_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 500 7 1 4 10 > Run_info/SYS_500_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 500 8 2 3 5 > Run_info/SYS_500_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 500 9 2 3 10 > Run_info/SYS_500_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 500 10 2 4 5 > Run_info/SYS_500_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 500 11 2 4 10 > Run_info/SYS_500_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 500 12 3 3 5 > Run_info/SYS_500_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 500 13 3 3 10 > Run_info/SYS_500_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 500 14 3 4 5 > Run_info/SYS_500_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 500 15 3 4 10 > Run_info/SYS_500_RUN_15.txt &
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
