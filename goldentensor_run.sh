#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 7 0 2 3 6 2 3 6 2 3 6 > Run_info/SYS_7_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 7 1 2 3 6 2 3 6 2 3 6 > Run_info/SYS_7_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 7 2 2 4 6 2 4 6 2 4 6 > Run_info/SYS_7_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 7 3 2 4 6 2 4 6 2 4 6 > Run_info/SYS_7_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 7 4 2 5 6 2 5 6 2 5 6 > Run_info/SYS_7_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 7 5 2 5 6 2 5 6 2 5 6 > Run_info/SYS_7_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 7 6 2 6 6 2 6 6 2 6 6 > Run_info/SYS_7_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 7 7 2 6 6 2 6 6 2 6 6 > Run_info/SYS_7_RUN_7.txt &
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
