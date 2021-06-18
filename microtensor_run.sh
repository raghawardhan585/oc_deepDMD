#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 0  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 1  1 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 2  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 3  2 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 4  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 5  3 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 6  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 7  4 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 8  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 9  5 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_9.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 10  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_10.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 11  6 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 12  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_12.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 13  7 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_13.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 14  8 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_14.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 701 15  8 5 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_15.txt &
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
