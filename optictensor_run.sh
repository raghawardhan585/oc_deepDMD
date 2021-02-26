#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 0  2 8 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 1  2 8 8 1 1 1 1 1 1 > Run_info/SYS_11_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 2  2 9 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 3  2 9 8 1 1 1 1 1 1 > Run_info/SYS_11_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 4  4 7 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 5  4 7 8 1 1 1 1 1 1 > Run_info/SYS_11_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 6  4 8 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 7  4 8 8 1 1 1 1 1 1 > Run_info/SYS_11_RUN_7.txt &
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
