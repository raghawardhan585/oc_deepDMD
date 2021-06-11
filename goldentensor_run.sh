#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 413 0 0 4 10 > Run_info/SYS_413_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 413 1 0 4 10 > Run_info/SYS_413_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 413 2 1 4 10 > Run_info/SYS_413_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 413 3 1 4 10 > Run_info/SYS_413_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 413 4 2 4 10 > Run_info/SYS_413_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 413 5 2 4 10 > Run_info/SYS_413_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 413 6 3 4 10 > Run_info/SYS_413_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 413 7 3 4 10 > Run_info/SYS_413_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 413 8 4 4 10 > Run_info/SYS_413_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 413 9 4 4 10 > Run_info/SYS_413_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 413 10 5 4 10 > Run_info/SYS_413_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 413 11 5 4 10 > Run_info/SYS_413_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 413 12 6 4 10 > Run_info/SYS_413_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 413 13 6 4 10 > Run_info/SYS_413_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 413 14 7 4 10 > Run_info/SYS_413_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 413 15 7 4 10 > Run_info/SYS_413_RUN_15.txt &
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
