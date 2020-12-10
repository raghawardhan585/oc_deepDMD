#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 0 15 8 22 3 5 6 9 4 11 > Run_info/SYS_27_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 1 15 8 22 3 5 6 9 4 11 > Run_info/SYS_27_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 2 18 8 25 3 5 6 9 4 14 > Run_info/SYS_27_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 3 18 8 25 3 5 6 9 4 14 > Run_info/SYS_27_RUN_3.txt &
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
