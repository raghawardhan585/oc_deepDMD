#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 22 0  3 9 9 1 1 1 1 1 1 > Run_info/SYS_22_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 22 1  3 9 12 1 1 1 1 1 1 > Run_info/SYS_22_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 22 2  3 9 15 1 1 1 1 1 1 > Run_info/SYS_22_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 22 3  3 12 9 1 1 1 1 1 1 > Run_info/SYS_22_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 22 4  4 6 12 1 1 1 1 1 1 > Run_info/SYS_22_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 22 5  4 6 15 1 1 1 1 1 1 > Run_info/SYS_22_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 22 6  4 9 9 1 1 1 1 1 1 > Run_info/SYS_22_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 22 7  4 9 12 1 1 1 1 1 1 > Run_info/SYS_22_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 22 8  5 3 15 1 1 1 1 1 1 > Run_info/SYS_22_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 22 9  5 6 9 1 1 1 1 1 1 > Run_info/SYS_22_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 22 10  5 6 12 1 1 1 1 1 1 > Run_info/SYS_22_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 22 11  5 6 15 1 1 1 1 1 1 > Run_info/SYS_22_RUN_11.txt &
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
