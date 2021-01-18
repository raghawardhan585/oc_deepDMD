#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 0  14 3 18 1 1 1 1 1 1 > Run_info/SYS_70_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 1  14 3 21 1 1 1 1 1 1 > Run_info/SYS_70_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 2  14 7 24 1 1 1 1 1 1 > Run_info/SYS_70_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 3  14 7 27 1 1 1 1 1 1 > Run_info/SYS_70_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 4  15 4 18 1 1 1 1 1 1 > Run_info/SYS_70_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 5  15 4 21 1 1 1 1 1 1 > Run_info/SYS_70_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 6  15 8 24 1 1 1 1 1 1 > Run_info/SYS_70_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 70 7  15 8 27 1 1 1 1 1 1 > Run_info/SYS_70_RUN_7.txt &
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
