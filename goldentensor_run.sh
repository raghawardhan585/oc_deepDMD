#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 91 0 3 3 3 > Run_info/SYS_91_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 91 1 3 3 6 > Run_info/SYS_91_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 91 2 3 4 3 > Run_info/SYS_91_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 91 3 3 4 6 > Run_info/SYS_91_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 4 3 3 3 > Run_info/SYS_91_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 91 5 3 3 6 > Run_info/SYS_91_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 91 6 3 4 3 > Run_info/SYS_91_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 91 7 3 4 6 > Run_info/SYS_91_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 91 8 3 3 3 > Run_info/SYS_91_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 91 9 3 3 6 > Run_info/SYS_91_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 91 10 3 4 3 > Run_info/SYS_91_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 91 11 3 4 6 > Run_info/SYS_91_RUN_11.txt &
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
