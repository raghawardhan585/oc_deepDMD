#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 53 0 10 3 20 > Run_info/SYS_53_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 53 1 10 3 25 > Run_info/SYS_53_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 53 2 10 3 30 > Run_info/SYS_53_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 53 3 10 4 10 > Run_info/SYS_53_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 53 4 11 3 20 > Run_info/SYS_53_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 53 5 11 3 25 > Run_info/SYS_53_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 53 6 11 3 30 > Run_info/SYS_53_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 53 7 11 4 10 > Run_info/SYS_53_RUN_7.txt &
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
