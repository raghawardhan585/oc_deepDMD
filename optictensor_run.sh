#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 0  4 5 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 1  4 5 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 2  4 6 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 3  4 6 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 4  4 8 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 5  4 8 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 6  4 8 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 7  4 8 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 8  5 6 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 9  5 6 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 10  5 7 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 11  5 7 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 12  6 5 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 13  6 5 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 14  6 5 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 15  6 5 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 16  6 7 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 17  6 7 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 18  6 8 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 19  6 8 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_19.txt &
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
