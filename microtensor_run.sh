#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 0 5 4 7 1 4 3 1 3 3 > Run_info/SYS_27_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 1 5 4 7 1 4 3 1 3 3 > Run_info/SYS_27_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 2 5 4 10 1 3 6 1 3 6 > Run_info/SYS_27_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 3 5 4 10 1 3 6 1 3 6 > Run_info/SYS_27_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 4 6 8 8 1 3 9 1 4 3 > Run_info/SYS_27_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 5 6 8 8 1 3 9 1 4 3 > Run_info/SYS_27_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 6 6 8 11 1 3 12 1 4 6 > Run_info/SYS_27_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 7 6 8 11 1 3 12 1 4 6 > Run_info/SYS_27_RUN_7.txt &
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
