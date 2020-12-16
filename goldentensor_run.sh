#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 52 0  4 9 9 1 1 1 1 1 1 > Run_info/SYS_52_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 52 1  4 9 12 1 1 1 1 1 1 > Run_info/SYS_52_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 52 2  4 9 15 1 1 1 1 1 1 > Run_info/SYS_52_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 52 3  4 12 9 1 1 1 1 1 1 > Run_info/SYS_52_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 52 4  5 6 12 1 1 1 1 1 1 > Run_info/SYS_52_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 52 5  5 6 15 1 1 1 1 1 1 > Run_info/SYS_52_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 52 6  5 9 9 1 1 1 1 1 1 > Run_info/SYS_52_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 52 7  5 9 12 1 1 1 1 1 1 > Run_info/SYS_52_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 52 8  6 3 15 1 1 1 1 1 1 > Run_info/SYS_52_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 52 9  6 6 9 1 1 1 1 1 1 > Run_info/SYS_52_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 52 10  6 6 12 1 1 1 1 1 1 > Run_info/SYS_52_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 52 11  6 6 15 1 1 1 1 1 1 > Run_info/SYS_52_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 52 12  7 3 9 1 1 1 1 1 1 > Run_info/SYS_52_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 52 13  7 3 12 1 1 1 1 1 1 > Run_info/SYS_52_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 52 14  7 3 15 1 1 1 1 1 1 > Run_info/SYS_52_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 52 15  7 6 9 1 1 1 1 1 1 > Run_info/SYS_52_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 52 16  7 12 12 1 1 1 1 1 1 > Run_info/SYS_52_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 52 17  7 12 15 1 1 1 1 1 1 > Run_info/SYS_52_RUN_17.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
