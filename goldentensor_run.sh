#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 404 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 404 1  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 404 2  1 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 404 3  1 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 404 4  2 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 404 5  2 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 404 6  3 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 404 7  3 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 404 8  4 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 404 9  4 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 404 10  5 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 404 11  5 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 404 12  6 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 404 13  6 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 404 14  7 3 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 404 15  7 4 20 1 1 1 1 1 1 0 > Run_info/SYS_404_RUN_15.txt &
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
