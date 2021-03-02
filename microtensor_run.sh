#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 0  4 3 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_0.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 1  4 3 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_1.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 2  4 3 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_2.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 3  4 3 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 4  5 4 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_4.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 5  5 4 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_5.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 6  5 4 12 1 1 1 1 1 1 > Run_info/SYS_91_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 7  5 4 15 1 1 1 1 1 1 > Run_info/SYS_91_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 8  6 3 8 1 1 1 1 1 1 > Run_info/SYS_91_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/cpu:0' 91 9  6 3 10 1 1 1 1 1 1 > Run_info/SYS_91_RUN_9.txt &
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
