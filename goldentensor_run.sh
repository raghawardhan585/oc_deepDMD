#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 80 0  1 1 1 1 8 6 1 1 1 > Run_info/SYS_80_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 80 1  1 1 1 1 8 9 1 1 1 > Run_info/SYS_80_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 80 2  1 1 1 2 3 6 1 1 1 > Run_info/SYS_80_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 80 3  1 1 1 2 3 9 1 1 1 > Run_info/SYS_80_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 80 4  1 1 1 3 3 6 1 1 1 > Run_info/SYS_80_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 80 5  1 1 1 3 3 9 1 1 1 > Run_info/SYS_80_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 80 6  1 1 1 3 4 6 1 1 1 > Run_info/SYS_80_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 80 7  1 1 1 3 4 9 1 1 1 > Run_info/SYS_80_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 80 8  1 1 1 4 4 6 1 1 1 > Run_info/SYS_80_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 80 9  1 1 1 4 4 9 1 1 1 > Run_info/SYS_80_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 80 10  1 1 1 4 7 6 1 1 1 > Run_info/SYS_80_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 80 11  1 1 1 4 7 9 1 1 1 > Run_info/SYS_80_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 80 12  1 1 1 5 7 6 1 1 1 > Run_info/SYS_80_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 80 13  1 1 1 5 7 9 1 1 1 > Run_info/SYS_80_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 80 14  1 1 1 5 8 6 1 1 1 > Run_info/SYS_80_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 80 15  1 1 1 5 8 9 1 1 1 > Run_info/SYS_80_RUN_15.txt &
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
