#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 0  4 5 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 1  4 5 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 2  4 7 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 3  4 7 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 4  5 6 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 5  5 6 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 6  5 8 5 1 1 1 1 1 1 > Run_info/SYS_21_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 7  5 8 6 1 1 1 1 1 1 > Run_info/SYS_21_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 8  6 7 3 1 1 1 1 1 1 > Run_info/SYS_21_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 21 9  6 7 4 1 1 1 1 1 1 > Run_info/SYS_21_RUN_9.txt &
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
