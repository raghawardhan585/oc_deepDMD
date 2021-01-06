#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 10 0  1 1 1 1 1 1 2 3 5 > Run_info/SYS_10_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 10 1  1 1 1 1 1 1 2 3 8 > Run_info/SYS_10_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 10 2  1 1 1 1 1 1 2 4 5 > Run_info/SYS_10_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 10 3  1 1 1 1 1 1 2 4 8 > Run_info/SYS_10_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 10 4  1 1 1 1 1 1 3 5 5 > Run_info/SYS_10_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 10 5  1 1 1 1 1 1 3 5 8 > Run_info/SYS_10_RUN_5.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
