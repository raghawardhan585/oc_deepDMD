#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 0  6 5 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 1  6 5 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 2  6 6 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 3  6 6 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 4  6 8 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 5  6 8 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 6  6 8 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 7  7 6 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_7.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 8  7 6 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 9  7 8 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 10  8 5 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_10.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 11  8 6 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_11.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 12  8 6 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 13  8 8 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_13.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
