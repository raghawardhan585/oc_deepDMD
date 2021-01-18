#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 70 0  10 3 20 1 1 1 1 1 1 > Run_info/SYS_70_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 70 1  10 3 25 1 1 1 1 1 1 > Run_info/SYS_70_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 70 2  10 4 10 1 1 1 1 1 1 > Run_info/SYS_70_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 70 3  10 4 15 1 1 1 1 1 1 > Run_info/SYS_70_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 70 4  11 4 10 1 1 1 1 1 1 > Run_info/SYS_70_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 70 5  11 4 15 1 1 1 1 1 1 > Run_info/SYS_70_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 70 6  11 4 20 1 1 1 1 1 1 > Run_info/SYS_70_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 70 7  11 4 25 1 1 1 1 1 1 > Run_info/SYS_70_RUN_7.txt &
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
