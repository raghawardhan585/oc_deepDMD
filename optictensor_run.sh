#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 0  30 3 30 1 1 1 1 1 1 0.00010000000000000002 > Run_info/SYS_304_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 1  30 3 30 1 1 1 1 1 1 0.000125 > Run_info/SYS_304_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 2  30 3 30 1 1 1 1 1 1 0.00015000000000000001 > Run_info/SYS_304_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 3  30 3 30 1 1 1 1 1 1 0.00017500000000000003 > Run_info/SYS_304_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 4  30 3 30 1 1 1 1 1 1 0.00035000000000000005 > Run_info/SYS_304_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 5  30 3 30 1 1 1 1 1 1 0.00037500000000000006 > Run_info/SYS_304_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 6  30 3 30 1 1 1 1 1 1 0.0004000000000000001 > Run_info/SYS_304_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 7  30 3 30 1 1 1 1 1 1 0.0004250000000000001 > Run_info/SYS_304_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 8  30 3 30 1 1 1 1 1 1 0.0006000000000000002 > Run_info/SYS_304_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 9  30 3 30 1 1 1 1 1 1 0.0006250000000000001 > Run_info/SYS_304_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 10  30 3 30 1 1 1 1 1 1 0.0006500000000000001 > Run_info/SYS_304_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 11  30 3 30 1 1 1 1 1 1 0.0006750000000000001 > Run_info/SYS_304_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 304 12  30 3 30 1 1 1 1 1 1 0.0008500000000000002 > Run_info/SYS_304_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 304 13  30 3 30 1 1 1 1 1 1 0.0008750000000000001 > Run_info/SYS_304_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 304 14  30 3 30 1 1 1 1 1 1 0.0009000000000000002 > Run_info/SYS_304_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 304 15  30 3 30 1 1 1 1 1 1 0.0009250000000000001 > Run_info/SYS_304_RUN_15.txt &
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
