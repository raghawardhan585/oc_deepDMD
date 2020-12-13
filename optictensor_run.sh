#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 51 0  4 4 15 1 1 1 1 1 1 > Run_info/SYS_51_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 51 1  4 4 20 1 1 1 1 1 1 > Run_info/SYS_51_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 51 2  4 4 25 1 1 1 1 1 1 > Run_info/SYS_51_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 51 3  4 5 5 1 1 1 1 1 1 > Run_info/SYS_51_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:1' 51 4  4 6 15 1 1 1 1 1 1 > Run_info/SYS_51_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 51 5  4 6 20 1 1 1 1 1 1 > Run_info/SYS_51_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 51 6  4 6 25 1 1 1 1 1 1 > Run_info/SYS_51_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:2' 51 7  5 5 15 1 1 1 1 1 1 > Run_info/SYS_51_RUN_7.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 51 8  5 5 20 1 1 1 1 1 1 > Run_info/SYS_51_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 51 9  5 6 25 1 1 1 1 1 1 > Run_info/SYS_51_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 51 10  6 4 15 1 1 1 1 1 1 > Run_info/SYS_51_RUN_10.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 51 11  6 5 20 1 1 1 1 1 1 > Run_info/SYS_51_RUN_11.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 51 12  6 5 25 1 1 1 1 1 1 > Run_info/SYS_51_RUN_12.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
