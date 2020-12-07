#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 25 0 20 4 22 1 5 3 4 3 4 > Run_info/SYS_25_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 25 1 20 4 22 1 5 3 4 3 4 > Run_info/SYS_25_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 25 2 20 4 27 1 5 6 4 3 7 > Run_info/SYS_25_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 25 3 20 4 27 1 5 6 4 3 7 > Run_info/SYS_25_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 25 4 21 4 23 1 5 9 4 4 4 > Run_info/SYS_25_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 25 5 21 4 23 1 5 9 4 4 4 > Run_info/SYS_25_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 25 6 21 4 28 1 5 12 4 4 7 > Run_info/SYS_25_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 25 7 21 4 28 1 5 12 4 4 7 > Run_info/SYS_25_RUN_7.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
