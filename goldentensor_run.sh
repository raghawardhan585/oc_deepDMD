#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 53 0 3 4 15 > Run_info/SYS_53_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 53 1 3 4 20 > Run_info/SYS_53_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 53 2 3 4 25 > Run_info/SYS_53_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 53 3 3 4 30 > Run_info/SYS_53_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 53 4 6 4 15 > Run_info/SYS_53_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 53 5 6 4 20 > Run_info/SYS_53_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 53 6 6 4 25 > Run_info/SYS_53_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 53 7 6 4 30 > Run_info/SYS_53_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 53 8 9 4 15 > Run_info/SYS_53_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 53 9 9 4 20 > Run_info/SYS_53_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 53 10 9 4 25 > Run_info/SYS_53_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 53 11 9 4 30 > Run_info/SYS_53_RUN_11.txt &
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
