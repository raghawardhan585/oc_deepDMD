#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 410 0 0 4 10 > Run_info/SYS_410_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 410 1 0 4 10 > Run_info/SYS_410_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 410 2 0 4 10 > Run_info/SYS_410_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 410 3 0 4 10 > Run_info/SYS_410_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 4 1 4 10 > Run_info/SYS_410_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 410 5 1 4 10 > Run_info/SYS_410_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 410 6 1 4 10 > Run_info/SYS_410_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 410 7 1 4 10 > Run_info/SYS_410_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 8 2 4 10 > Run_info/SYS_410_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 410 9 2 4 10 > Run_info/SYS_410_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 410 10 2 4 10 > Run_info/SYS_410_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 410 11 2 4 10 > Run_info/SYS_410_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 12 3 4 10 > Run_info/SYS_410_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 410 13 3 4 10 > Run_info/SYS_410_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 410 14 3 4 10 > Run_info/SYS_410_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 410 15 3 4 10 > Run_info/SYS_410_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 16 4 4 10 > Run_info/SYS_410_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 410 17 4 4 10 > Run_info/SYS_410_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 410 18 4 4 10 > Run_info/SYS_410_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 410 19 4 4 10 > Run_info/SYS_410_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 20 5 4 10 > Run_info/SYS_410_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 410 21 5 4 10 > Run_info/SYS_410_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 410 22 5 4 10 > Run_info/SYS_410_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 410 23 5 4 10 > Run_info/SYS_410_RUN_23.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 24 6 4 10 > Run_info/SYS_410_RUN_24.txt &
python3 deepDMD.py '/gpu:1' 410 25 6 4 10 > Run_info/SYS_410_RUN_25.txt &
python3 deepDMD.py '/gpu:2' 410 26 6 4 10 > Run_info/SYS_410_RUN_26.txt &
python3 deepDMD.py '/gpu:3' 410 27 6 4 10 > Run_info/SYS_410_RUN_27.txt &
wait 
python3 deepDMD.py '/gpu:0' 410 28 7 4 10 > Run_info/SYS_410_RUN_28.txt &
python3 deepDMD.py '/gpu:1' 410 29 7 4 10 > Run_info/SYS_410_RUN_29.txt &
python3 deepDMD.py '/gpu:2' 410 30 7 4 10 > Run_info/SYS_410_RUN_30.txt &
python3 deepDMD.py '/gpu:3' 410 31 7 4 10 > Run_info/SYS_410_RUN_31.txt &
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
