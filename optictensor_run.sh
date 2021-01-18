#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 70 0  14 3 24 1 1 1 1 1 1 > Run_info/SYS_70_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 70 1  14 3 27 1 1 1 1 1 1 > Run_info/SYS_70_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 70 2  14 4 18 1 1 1 1 1 1 > Run_info/SYS_70_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 70 3  14 4 21 1 1 1 1 1 1 > Run_info/SYS_70_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 70 4  14 8 18 1 1 1 1 1 1 > Run_info/SYS_70_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 70 5  14 8 21 1 1 1 1 1 1 > Run_info/SYS_70_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 70 6  14 8 24 1 1 1 1 1 1 > Run_info/SYS_70_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 70 7  14 8 27 1 1 1 1 1 1 > Run_info/SYS_70_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 70 8  15 4 24 1 1 1 1 1 1 > Run_info/SYS_70_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 70 9  15 4 27 1 1 1 1 1 1 > Run_info/SYS_70_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 70 10  15 7 18 1 1 1 1 1 1 > Run_info/SYS_70_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 70 11  15 7 21 1 1 1 1 1 1 > Run_info/SYS_70_RUN_11.txt &
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
