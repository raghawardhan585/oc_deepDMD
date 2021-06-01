#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 0  0 1 0 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 1  0 1 0 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 2  2 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 3  2 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 4  4 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 5  4 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 6  8 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 7  8 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 8  10 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 9  10 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 10  15 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 11  15 3 15 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_11.txt &
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
