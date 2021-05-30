#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 0  30 3 30 1 1 1 1 1 1 5e-05 > Run_info/SYS_304_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 1  30 3 30 1 1 1 1 1 1 7.500000000000001e-05 > Run_info/SYS_304_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 2  30 3 30 1 1 1 1 1 1 0.0003000000000000001 > Run_info/SYS_304_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 3  30 3 30 1 1 1 1 1 1 0.0003250000000000001 > Run_info/SYS_304_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 4  30 3 30 1 1 1 1 1 1 0.0005500000000000001 > Run_info/SYS_304_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 5  30 3 30 1 1 1 1 1 1 0.0005750000000000001 > Run_info/SYS_304_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 6  30 3 30 1 1 1 1 1 1 0.0008000000000000001 > Run_info/SYS_304_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 304 7  30 3 30 1 1 1 1 1 1 0.0008250000000000002 > Run_info/SYS_304_RUN_7.txt &
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
