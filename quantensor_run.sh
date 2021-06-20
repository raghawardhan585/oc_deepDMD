#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 0  0 1 0 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 1  0 1 0 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 2  1 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 3  1 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 4  2 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 5  2 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 6  3 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 7  3 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 8  4 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 9  4 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_9.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 10  5 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_10.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 11  5 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 12  6 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_12.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 13  6 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_13.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 14  7 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_14.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 704 15  7 3 10 1 1 1 1 1 1 0.005 > Run_info/SYS_704_RUN_15.txt &
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
