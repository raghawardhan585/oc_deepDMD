#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 0  4 4 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 1  4 4 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 2  4 6 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 3  4 6 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 4  5 5 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 5  5 5 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 6  6 4 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 7  6 4 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 8  6 6 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 27 9  6 6 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_9.txt &
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
