#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 30 0  4 3 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 30 1  4 3 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 30 2  4 6 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 30 3  4 6 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 30 4  5 9 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 30 5  5 9 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 30 6  6 3 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 30 7  6 3 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 30 8  7 6 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 30 9  7 6 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 30 10  7 9 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 30 11  7 9 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 30 12  9 3 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 30 13  9 3 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 30 14  9 6 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 30 15  9 6 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_15.txt &
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
