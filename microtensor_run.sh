#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/cpu:0' 53 0 5 6 5 > Run_info/SYS_53_RUN_0.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 1 5 6 6 > Run_info/SYS_53_RUN_1.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 2 5 9 11 > Run_info/SYS_53_RUN_2.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 3 5 9 12 > Run_info/SYS_53_RUN_3.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 4 8 7 5 > Run_info/SYS_53_RUN_4.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 5 8 7 6 > Run_info/SYS_53_RUN_5.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 6 8 10 11 > Run_info/SYS_53_RUN_6.txt &
wait 
python3 deepDMD.py '/cpu:0' 53 7 8 10 12 > Run_info/SYS_53_RUN_7.txt &
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
