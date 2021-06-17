#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 700 0 0 3 10 > Run_info/SYS_700_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 700 1 0 3 10 > Run_info/SYS_700_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 700 2 0 4 10 > Run_info/SYS_700_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 700 3 0 4 10 > Run_info/SYS_700_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 700 4 1 3 10 > Run_info/SYS_700_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 700 5 1 3 10 > Run_info/SYS_700_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 700 6 1 4 10 > Run_info/SYS_700_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 700 7 1 4 10 > Run_info/SYS_700_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 700 8 2 3 10 > Run_info/SYS_700_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 700 9 2 3 10 > Run_info/SYS_700_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 700 10 2 4 10 > Run_info/SYS_700_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 700 11 2 4 10 > Run_info/SYS_700_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 700 12 3 3 10 > Run_info/SYS_700_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 700 13 3 3 10 > Run_info/SYS_700_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 700 14 3 4 10 > Run_info/SYS_700_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 700 15 3 4 10 > Run_info/SYS_700_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 700 16 4 3 10 > Run_info/SYS_700_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 700 17 4 3 10 > Run_info/SYS_700_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 700 18 4 4 10 > Run_info/SYS_700_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 700 19 4 4 10 > Run_info/SYS_700_RUN_19.txt &
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
