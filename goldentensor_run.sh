#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 0  4 4 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 1  4 4 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 2  4 3 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 3  4 3 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 4  5 3 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 5  5 3 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 6  5 3 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 7  5 3 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 8  5 3 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 9  5 3 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 10  5 4 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 11  5 4 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 12  6 4 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 13  6 4 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 14  6 4 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 15  6 4 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 16  6 4 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 17  6 4 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_17.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
