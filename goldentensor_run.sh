#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 100 0  10 4 50 1 1 1 1 1 1 > Run_info/SYS_100_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 100 1  10 4 60 1 1 1 1 1 1 > Run_info/SYS_100_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 100 2  15 3 30 1 1 1 1 1 1 > Run_info/SYS_100_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 100 3  15 3 40 1 1 1 1 1 1 > Run_info/SYS_100_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 100 4  20 3 30 1 1 1 1 1 1 > Run_info/SYS_100_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 100 5  20 3 40 1 1 1 1 1 1 > Run_info/SYS_100_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 100 6  20 3 50 1 1 1 1 1 1 > Run_info/SYS_100_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 100 7  20 3 60 1 1 1 1 1 1 > Run_info/SYS_100_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 100 8  25 3 50 1 1 1 1 1 1 > Run_info/SYS_100_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 100 9  25 3 60 1 1 1 1 1 1 > Run_info/SYS_100_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 100 10  25 4 30 1 1 1 1 1 1 > Run_info/SYS_100_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 100 11  25 4 40 1 1 1 1 1 1 > Run_info/SYS_100_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 100 12  30 4 30 1 1 1 1 1 1 > Run_info/SYS_100_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 100 13  30 4 40 1 1 1 1 1 1 > Run_info/SYS_100_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 100 14  30 4 50 1 1 1 1 1 1 > Run_info/SYS_100_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 100 15  30 4 60 1 1 1 1 1 1 > Run_info/SYS_100_RUN_15.txt &
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
