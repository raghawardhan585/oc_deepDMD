#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 28 0  4 8 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 28 1  4 8 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 28 2  4 8 25 1 1 1 1 1 1 > Run_info/SYS_28_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 28 3  5 7 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_3.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 28 4  5 7 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 28 5  5 7 25 1 1 1 1 1 1 > Run_info/SYS_28_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 28 6  6 6 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 28 7  6 6 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_7.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 28 8  6 6 25 1 1 1 1 1 1 > Run_info/SYS_28_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 28 9  6 9 15 1 1 1 1 1 1 > Run_info/SYS_28_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 28 10  6 9 20 1 1 1 1 1 1 > Run_info/SYS_28_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 28 11  6 9 25 1 1 1 1 1 1 > Run_info/SYS_28_RUN_11.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
