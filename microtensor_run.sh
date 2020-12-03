#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 0 9 6 9 1 4 3 3 3 3 > Run_info/SYS_23_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 1 9 6 9 1 4 3 3 3 3 > Run_info/SYS_23_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 2 9 7 9 1 3 6 3 3 6 > Run_info/SYS_23_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 3 9 7 9 1 3 6 3 3 6 > Run_info/SYS_23_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 4 9 9 15 1 3 9 3 4 3 > Run_info/SYS_23_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 5 9 9 15 1 3 9 3 4 3 > Run_info/SYS_23_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 6 9 9 18 1 3 12 3 4 6 > Run_info/SYS_23_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 23 7 9 9 18 1 3 12 3 4 6 > Run_info/SYS_23_RUN_7.txt &
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
