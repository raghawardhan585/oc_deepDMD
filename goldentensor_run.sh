#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 600 0 0 4 5 > Run_info/SYS_600_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 600 1 0 4 10 > Run_info/SYS_600_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 600 2 1 4 5 > Run_info/SYS_600_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 600 3 1 4 10 > Run_info/SYS_600_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 600 4 2 4 5 > Run_info/SYS_600_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 600 5 2 4 10 > Run_info/SYS_600_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 600 6 3 4 5 > Run_info/SYS_600_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 600 7 3 4 10 > Run_info/SYS_600_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 600 8 4 4 5 > Run_info/SYS_600_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 600 9 4 4 10 > Run_info/SYS_600_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 600 10 5 4 5 > Run_info/SYS_600_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 600 11 5 4 10 > Run_info/SYS_600_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 600 12 6 4 5 > Run_info/SYS_600_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 600 13 6 4 10 > Run_info/SYS_600_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 600 14 7 4 5 > Run_info/SYS_600_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 600 15 7 4 10 > Run_info/SYS_600_RUN_15.txt &
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
