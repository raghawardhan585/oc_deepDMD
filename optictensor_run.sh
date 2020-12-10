#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 0 15 14 22 3 5 6 13 4 15 > Run_info/SYS_27_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 1 15 14 22 3 5 6 13 4 15 > Run_info/SYS_27_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 2 18 14 25 3 5 6 13 4 18 > Run_info/SYS_27_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 3 18 14 25 3 5 6 13 4 18 > Run_info/SYS_27_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 4 21 14 28 3 5 6 14 4 16 > Run_info/SYS_27_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 5 21 14 28 3 5 6 14 4 16 > Run_info/SYS_27_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 6 24 14 31 3 5 6 14 4 19 > Run_info/SYS_27_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 7 24 14 31 3 5 6 14 4 19 > Run_info/SYS_27_RUN_7.txt &
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
