#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 0  3 4 5 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 1  3 4 5 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 2  3 4 5 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 3  3 4 5 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 4  3 4 10 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 5  3 4 10 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 6  3 4 10 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 7  3 4 10 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 8  3 4 15 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 9  3 4 15 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 10  3 4 15 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 11  3 4 15 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 12  3 4 20 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 13  3 4 20 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 14  3 4 20 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 15  3 4 20 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_15.txt &
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
