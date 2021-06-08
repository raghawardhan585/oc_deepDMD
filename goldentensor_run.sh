#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 501 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_501_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 501 1  0 1 0 1 1 1 1 1 1 5e-06 > Run_info/SYS_501_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 501 2  0 1 0 1 1 1 1 1 1 1e-05 > Run_info/SYS_501_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 501 3  0 1 0 1 1 1 1 1 1 5e-05 > Run_info/SYS_501_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 501 4  0 1 0 1 1 1 1 1 1 0.0001 > Run_info/SYS_501_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 501 5  0 1 0 1 1 1 1 1 1 0.0005 > Run_info/SYS_501_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 501 6  0 1 0 1 1 1 1 1 1 0.001 > Run_info/SYS_501_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 501 7  0 1 0 1 1 1 1 1 1 0.005 > Run_info/SYS_501_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 501 8  0 1 0 1 1 1 1 1 1 0.01 > Run_info/SYS_501_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 501 9  0 1 0 1 1 1 1 1 1 0.05 > Run_info/SYS_501_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 501 10  0 1 0 1 1 1 1 1 1 0.1 > Run_info/SYS_501_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 501 11  0 1 0 1 1 1 1 1 1 0.5 > Run_info/SYS_501_RUN_11.txt &
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
