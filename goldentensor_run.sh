#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 25 0 27 4 31 1 4 3 2 3 3 > Run_info/SYS_25_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 25 1 27 4 31 1 4 3 2 3 3 > Run_info/SYS_25_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 25 2 27 4 34 1 4 6 2 3 6 > Run_info/SYS_25_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 25 3 27 4 34 1 4 6 2 3 6 > Run_info/SYS_25_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 25 4 28 4 32 1 4 9 2 4 3 > Run_info/SYS_25_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 25 5 28 4 32 1 4 9 2 4 3 > Run_info/SYS_25_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 25 6 28 4 35 1 4 12 2 4 6 > Run_info/SYS_25_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 25 7 28 4 35 1 4 12 2 4 6 > Run_info/SYS_25_RUN_7.txt &
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
