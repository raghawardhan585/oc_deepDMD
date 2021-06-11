#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 409 0 0 4 10 > Run_info/SYS_409_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 409 1 0 4 10 > Run_info/SYS_409_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 409 2 0 4 10 > Run_info/SYS_409_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 409 3 0 4 10 > Run_info/SYS_409_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 4 0 4 10 > Run_info/SYS_409_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 409 5 1 4 10 > Run_info/SYS_409_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 409 6 1 4 10 > Run_info/SYS_409_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 409 7 1 4 10 > Run_info/SYS_409_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 8 1 4 10 > Run_info/SYS_409_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 409 9 1 4 10 > Run_info/SYS_409_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 409 10 2 4 10 > Run_info/SYS_409_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 409 11 2 4 10 > Run_info/SYS_409_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 12 2 4 10 > Run_info/SYS_409_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 409 13 2 4 10 > Run_info/SYS_409_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 409 14 2 4 10 > Run_info/SYS_409_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 409 15 3 4 10 > Run_info/SYS_409_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 16 3 4 10 > Run_info/SYS_409_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 409 17 3 4 10 > Run_info/SYS_409_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 409 18 3 4 10 > Run_info/SYS_409_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 409 19 3 4 10 > Run_info/SYS_409_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 20 4 4 10 > Run_info/SYS_409_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 409 21 4 4 10 > Run_info/SYS_409_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 409 22 4 4 10 > Run_info/SYS_409_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 409 23 4 4 10 > Run_info/SYS_409_RUN_23.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 24 4 4 10 > Run_info/SYS_409_RUN_24.txt &
python3 deepDMD.py '/gpu:1' 409 25 5 4 10 > Run_info/SYS_409_RUN_25.txt &
python3 deepDMD.py '/gpu:2' 409 26 5 4 10 > Run_info/SYS_409_RUN_26.txt &
python3 deepDMD.py '/gpu:3' 409 27 5 4 10 > Run_info/SYS_409_RUN_27.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 28 5 4 10 > Run_info/SYS_409_RUN_28.txt &
python3 deepDMD.py '/gpu:1' 409 29 5 4 10 > Run_info/SYS_409_RUN_29.txt &
python3 deepDMD.py '/gpu:2' 409 30 6 4 10 > Run_info/SYS_409_RUN_30.txt &
python3 deepDMD.py '/gpu:3' 409 31 6 4 10 > Run_info/SYS_409_RUN_31.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 32 6 4 10 > Run_info/SYS_409_RUN_32.txt &
python3 deepDMD.py '/gpu:1' 409 33 6 4 10 > Run_info/SYS_409_RUN_33.txt &
python3 deepDMD.py '/gpu:2' 409 34 6 4 10 > Run_info/SYS_409_RUN_34.txt &
python3 deepDMD.py '/gpu:3' 409 35 7 4 10 > Run_info/SYS_409_RUN_35.txt &
wait 
python3 deepDMD.py '/gpu:0' 409 36 7 4 10 > Run_info/SYS_409_RUN_36.txt &
python3 deepDMD.py '/gpu:1' 409 37 7 4 10 > Run_info/SYS_409_RUN_37.txt &
python3 deepDMD.py '/gpu:2' 409 38 7 4 10 > Run_info/SYS_409_RUN_38.txt &
python3 deepDMD.py '/gpu:3' 409 39 7 4 10 > Run_info/SYS_409_RUN_39.txt &
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
