#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 702 0 0 4 10 > Run_info/SYS_702_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 702 1 0 4 10 > Run_info/SYS_702_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 702 2 1 4 10 > Run_info/SYS_702_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 702 3 1 4 10 > Run_info/SYS_702_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 702 4 2 4 10 > Run_info/SYS_702_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 702 5 2 4 10 > Run_info/SYS_702_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 702 6 3 4 10 > Run_info/SYS_702_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 702 7 3 4 10 > Run_info/SYS_702_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 702 8 4 4 10 > Run_info/SYS_702_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 702 9 4 4 10 > Run_info/SYS_702_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 702 10 5 4 10 > Run_info/SYS_702_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 702 11 5 4 10 > Run_info/SYS_702_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 702 12 6 4 10 > Run_info/SYS_702_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 702 13 6 4 10 > Run_info/SYS_702_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 702 14 7 4 10 > Run_info/SYS_702_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 702 15 7 4 10 > Run_info/SYS_702_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 702 16 8 4 10 > Run_info/SYS_702_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 702 17 8 4 10 > Run_info/SYS_702_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 702 18 9 4 10 > Run_info/SYS_702_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 702 19 9 4 10 > Run_info/SYS_702_RUN_19.txt &
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
