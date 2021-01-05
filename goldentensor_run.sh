#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 10 0  1 5 3 1 1 1 1 1 1 > Run_info/SYS_10_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 10 1  1 5 5 1 1 1 1 1 1 > Run_info/SYS_10_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 10 2  1 5 8 1 1 1 1 1 1 > Run_info/SYS_10_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 10 3  2 3 3 1 1 1 1 1 1 > Run_info/SYS_10_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 10 4  2 5 5 1 1 1 1 1 1 > Run_info/SYS_10_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 10 5  2 5 8 1 1 1 1 1 1 > Run_info/SYS_10_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 10 6  3 3 3 1 1 1 1 1 1 > Run_info/SYS_10_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 10 7  3 3 5 1 1 1 1 1 1 > Run_info/SYS_10_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 10 8  3 5 8 1 1 1 1 1 1 > Run_info/SYS_10_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 10 9  4 3 3 1 1 1 1 1 1 > Run_info/SYS_10_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 10 10  4 3 5 1 1 1 1 1 1 > Run_info/SYS_10_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 10 11  4 3 8 1 1 1 1 1 1 > Run_info/SYS_10_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 10 12  5 3 3 1 1 1 1 1 1 > Run_info/SYS_10_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 10 13  5 3 5 1 1 1 1 1 1 > Run_info/SYS_10_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 10 14  5 3 8 1 1 1 1 1 1 > Run_info/SYS_10_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 10 15  5 4 3 1 1 1 1 1 1 > Run_info/SYS_10_RUN_15.txt &
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
