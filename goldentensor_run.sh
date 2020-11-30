#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 0 4 3 6 1 3 6 1 3 6 > Run_info/SYS_21_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 21 1 4 3 6 1 3 6 1 3 6 > Run_info/SYS_21_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 2 4 4 6 1 4 6 1 4 6 > Run_info/SYS_21_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 21 3 4 4 6 1 4 6 1 4 6 > Run_info/SYS_21_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 4 4 5 6 1 5 6 1 5 6 > Run_info/SYS_21_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 21 5 4 5 6 1 5 6 1 5 6 > Run_info/SYS_21_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 6 4 6 6 1 6 6 1 6 6 > Run_info/SYS_21_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 21 7 4 6 6 1 6 6 1 6 6 > Run_info/SYS_21_RUN_7.txt &
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
