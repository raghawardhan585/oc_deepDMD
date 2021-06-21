#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 704 0 8 4 10 > Run_info/SYS_704_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 704 1 8 4 10 > Run_info/SYS_704_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 704 2 8 4 10 > Run_info/SYS_704_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 704 3 8 4 10 > Run_info/SYS_704_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 704 4 9 4 10 > Run_info/SYS_704_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 704 5 9 4 10 > Run_info/SYS_704_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 704 6 9 4 10 > Run_info/SYS_704_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 704 7 9 4 10 > Run_info/SYS_704_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 704 8 10 4 10 > Run_info/SYS_704_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 704 9 10 4 10 > Run_info/SYS_704_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 704 10 10 4 10 > Run_info/SYS_704_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 704 11 10 4 10 > Run_info/SYS_704_RUN_11.txt &
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
