#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 26 0 9 4 11 1 5 3 4 3 4 > Run_info/SYS_26_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 26 1 9 4 11 1 5 3 4 3 4 > Run_info/SYS_26_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 26 2 9 4 14 1 5 6 4 3 7 > Run_info/SYS_26_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 26 3 9 4 14 1 5 6 4 3 7 > Run_info/SYS_26_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 26 4 10 4 12 1 5 9 4 4 4 > Run_info/SYS_26_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 26 5 10 4 12 1 5 9 4 4 4 > Run_info/SYS_26_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 26 6 10 4 15 1 5 12 4 4 7 > Run_info/SYS_26_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 26 7 10 4 15 1 5 12 4 4 7 > Run_info/SYS_26_RUN_7.txt &
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
