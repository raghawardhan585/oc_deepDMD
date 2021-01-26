#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 0  5 3 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 1  5 3 20 1 1 1 1 1 1 > Run_info/SYS_101_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 2  10 3 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 3  10 3 20 1 1 1 1 1 1 > Run_info/SYS_101_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 4  15 3 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 5  15 3 20 1 1 1 1 1 1 > Run_info/SYS_101_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 6  20 3 10 1 1 1 1 1 1 > Run_info/SYS_101_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 101 7  20 3 20 1 1 1 1 1 1 > Run_info/SYS_101_RUN_7.txt &
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
