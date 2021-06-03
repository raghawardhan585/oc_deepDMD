#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 401 0  1 1 1 1 3 5 1 1 1 0 > Run_info/SYS_401_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 401 1  1 1 1 1 3 10 1 1 1 0 > Run_info/SYS_401_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 401 2  1 1 1 1 4 5 1 1 1 0 > Run_info/SYS_401_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 401 3  1 1 1 1 4 10 1 1 1 0 > Run_info/SYS_401_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 401 4  1 1 1 1 5 5 1 1 1 0 > Run_info/SYS_401_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 401 5  1 1 1 1 5 10 1 1 1 0 > Run_info/SYS_401_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 401 6  1 1 1 1 6 5 1 1 1 0 > Run_info/SYS_401_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 401 7  1 1 1 1 6 10 1 1 1 0 > Run_info/SYS_401_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 401 8  1 1 1 2 3 5 1 1 1 0 > Run_info/SYS_401_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 401 9  1 1 1 2 3 10 1 1 1 0 > Run_info/SYS_401_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 401 10  1 1 1 2 4 5 1 1 1 0 > Run_info/SYS_401_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 401 11  1 1 1 2 4 10 1 1 1 0 > Run_info/SYS_401_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 401 12  1 1 1 2 5 5 1 1 1 0 > Run_info/SYS_401_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 401 13  1 1 1 2 5 10 1 1 1 0 > Run_info/SYS_401_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 401 14  1 1 1 2 6 5 1 1 1 0 > Run_info/SYS_401_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 401 15  1 1 1 2 6 10 1 1 1 0 > Run_info/SYS_401_RUN_15.txt &
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
