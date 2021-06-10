#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 601 0  4 7 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 601 1  4 8 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 601 2  5 7 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 601 3  5 8 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 601 4  6 7 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 601 5  6 8 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 601 6  7 7 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 601 7  7 8 10 1 1 1 1 1 1 0 > Run_info/SYS_601_RUN_7.txt &
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
