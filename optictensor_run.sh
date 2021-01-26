#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 101 0  5 3 30 1 1 1 1 1 1 > Run_info/SYS_101_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 101 1  5 3 40 1 1 1 1 1 1 > Run_info/SYS_101_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 101 2  5 3 50 1 1 1 1 1 1 > Run_info/SYS_101_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 101 3  5 4 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 101 4  10 3 30 1 1 1 1 1 1 > Run_info/SYS_101_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 101 5  10 3 40 1 1 1 1 1 1 > Run_info/SYS_101_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 101 6  10 3 50 1 1 1 1 1 1 > Run_info/SYS_101_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 101 7  10 4 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 101 8  15 3 30 1 1 1 1 1 1 > Run_info/SYS_101_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 101 9  15 3 40 1 1 1 1 1 1 > Run_info/SYS_101_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 101 10  15 3 50 1 1 1 1 1 1 > Run_info/SYS_101_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 101 11  15 4 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 101 12  20 3 30 1 1 1 1 1 1 > Run_info/SYS_101_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 101 13  20 3 40 1 1 1 1 1 1 > Run_info/SYS_101_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 101 14  20 3 50 1 1 1 1 1 1 > Run_info/SYS_101_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 101 15  20 4 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_15.txt &
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
