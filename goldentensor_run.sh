#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 28 0  3 7 9 1 1 1 1 1 1 > Run_info/SYS_28_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 28 1  3 7 12 1 1 1 1 1 1 > Run_info/SYS_28_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 28 2  3 10 3 1 1 1 1 1 1 > Run_info/SYS_28_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 28 3  3 10 6 1 1 1 1 1 1 > Run_info/SYS_28_RUN_3.txt &
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
