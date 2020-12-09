#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 31 0 23 4 31 3 4 3 4 3 4 > Run_info/SYS_31_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 31 1 23 4 31 3 4 3 4 3 4 > Run_info/SYS_31_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 31 2 23 4 31 3 4 6 4 3 7 > Run_info/SYS_31_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 31 3 23 4 31 3 4 6 4 3 7 > Run_info/SYS_31_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 31 4 23 4 31 3 5 3 4 4 4 > Run_info/SYS_31_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 31 5 23 4 31 3 5 3 4 4 4 > Run_info/SYS_31_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 31 6 23 4 31 3 5 6 4 4 7 > Run_info/SYS_31_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 31 7 23 4 31 3 5 6 4 4 7 > Run_info/SYS_31_RUN_7.txt &
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
