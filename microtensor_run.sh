#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 0  4 6 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 1  4 6 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 2  4 9 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 3  4 9 25 1 1 1 1 1 1 > Run_info/SYS_28_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 4  5 8 25 1 1 1 1 1 1 > Run_info/SYS_28_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 5  5 9 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 6  6 8 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 28 7  6 8 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_7.txt &
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
