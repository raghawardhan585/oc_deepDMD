#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 414 0 0 3 10 > Run_info/SYS_414_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 414 1 0 4 10 > Run_info/SYS_414_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 414 2 0 3 10 > Run_info/SYS_414_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 414 3 0 4 10 > Run_info/SYS_414_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 4 1 3 10 > Run_info/SYS_414_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 414 5 1 4 10 > Run_info/SYS_414_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 414 6 1 3 10 > Run_info/SYS_414_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 414 7 1 4 10 > Run_info/SYS_414_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 8 2 3 10 > Run_info/SYS_414_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 414 9 2 4 10 > Run_info/SYS_414_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 414 10 2 3 10 > Run_info/SYS_414_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 414 11 2 4 10 > Run_info/SYS_414_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 12 3 3 10 > Run_info/SYS_414_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 414 13 3 4 10 > Run_info/SYS_414_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 414 14 3 3 10 > Run_info/SYS_414_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 414 15 3 4 10 > Run_info/SYS_414_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 16 4 3 10 > Run_info/SYS_414_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 414 17 4 4 10 > Run_info/SYS_414_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 414 18 4 3 10 > Run_info/SYS_414_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 414 19 4 4 10 > Run_info/SYS_414_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 20 5 3 10 > Run_info/SYS_414_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 414 21 5 4 10 > Run_info/SYS_414_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 414 22 5 3 10 > Run_info/SYS_414_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 414 23 5 4 10 > Run_info/SYS_414_RUN_23.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 24 6 3 10 > Run_info/SYS_414_RUN_24.txt &
python3 deepDMD.py '/gpu:1' 414 25 6 4 10 > Run_info/SYS_414_RUN_25.txt &
python3 deepDMD.py '/gpu:2' 414 26 6 3 10 > Run_info/SYS_414_RUN_26.txt &
python3 deepDMD.py '/gpu:3' 414 27 6 4 10 > Run_info/SYS_414_RUN_27.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 28 7 3 10 > Run_info/SYS_414_RUN_28.txt &
python3 deepDMD.py '/gpu:1' 414 29 7 4 10 > Run_info/SYS_414_RUN_29.txt &
python3 deepDMD.py '/gpu:2' 414 30 7 3 10 > Run_info/SYS_414_RUN_30.txt &
python3 deepDMD.py '/gpu:3' 414 31 7 4 10 > Run_info/SYS_414_RUN_31.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 32 8 3 10 > Run_info/SYS_414_RUN_32.txt &
python3 deepDMD.py '/gpu:1' 414 33 8 4 10 > Run_info/SYS_414_RUN_33.txt &
python3 deepDMD.py '/gpu:2' 414 34 8 3 10 > Run_info/SYS_414_RUN_34.txt &
python3 deepDMD.py '/gpu:3' 414 35 8 4 10 > Run_info/SYS_414_RUN_35.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 36 9 3 10 > Run_info/SYS_414_RUN_36.txt &
python3 deepDMD.py '/gpu:1' 414 37 9 4 10 > Run_info/SYS_414_RUN_37.txt &
python3 deepDMD.py '/gpu:2' 414 38 9 3 10 > Run_info/SYS_414_RUN_38.txt &
python3 deepDMD.py '/gpu:3' 414 39 9 4 10 > Run_info/SYS_414_RUN_39.txt &
wait 
python3 deepDMD.py '/gpu:0' 414 40 10 3 10 > Run_info/SYS_414_RUN_40.txt &
python3 deepDMD.py '/gpu:1' 414 41 10 4 10 > Run_info/SYS_414_RUN_41.txt &
python3 deepDMD.py '/gpu:2' 414 42 10 3 10 > Run_info/SYS_414_RUN_42.txt &
python3 deepDMD.py '/gpu:3' 414 43 10 4 10 > Run_info/SYS_414_RUN_43.txt &
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
