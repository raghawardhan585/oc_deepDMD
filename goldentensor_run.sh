#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 0  2 3 10 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 1  2 3 15 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 2  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 3  2 4 15 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 4  5 3 10 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 5  5 3 15 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 6  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 7  5 4 15 1 1 1 1 1 1 0 > Run_info/SYS_402_RUN_7.txt &
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
