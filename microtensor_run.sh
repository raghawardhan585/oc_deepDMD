#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 0  3 3 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 1  3 3 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 2  4 9 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 3  4 9 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 4  6 6 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 5  6 6 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 6  8 3 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 7  8 3 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 8  9 9 9 1 1 1 1 1 1 > Run_info/SYS_30_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 30 9  9 9 12 1 1 1 1 1 1 > Run_info/SYS_30_RUN_9.txt &
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
