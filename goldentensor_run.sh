#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 0  5 3 10 1 1 1 1 1 1 2e-05 > Run_info/SYS_304_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 1  5 3 10 1 1 1 1 1 1 1e-05 > Run_info/SYS_304_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 2  5 4 10 1 1 1 1 1 1 8e-05 > Run_info/SYS_304_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 3  5 4 10 1 1 1 1 1 1 7e-05 > Run_info/SYS_304_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 4  10 3 10 1 1 1 1 1 1 8e-05 > Run_info/SYS_304_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 5  10 3 10 1 1 1 1 1 1 7e-05 > Run_info/SYS_304_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 6  10 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_304_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 7  10 3 10 1 1 1 1 1 1 5e-05 > Run_info/SYS_304_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 8  10 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_304_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 9  10 4 10 1 1 1 1 1 1 5e-05 > Run_info/SYS_304_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 10  10 4 10 1 1 1 1 1 1 4e-05 > Run_info/SYS_304_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 11  10 4 10 1 1 1 1 1 1 3e-05 > Run_info/SYS_304_RUN_11.txt &
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
