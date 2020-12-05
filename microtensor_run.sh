#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 24 0 9 4 9 1 4 3 1 3 3 > Run_info/SYS_24_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 24 1 9 4 9 1 4 3 1 3 3 > Run_info/SYS_24_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 24 2 6 4 9 1 3 6 1 3 6 > Run_info/SYS_24_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 24 3 6 4 9 1 3 6 1 3 6 > Run_info/SYS_24_RUN_3.txt &
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
