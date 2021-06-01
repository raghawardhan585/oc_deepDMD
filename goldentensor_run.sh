#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 308 0  0 1 0 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 308 1  0 1 0 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 308 2  2 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 308 3  2 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 308 4  4 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 308 5  4 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 308 6  6 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 308 7  6 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 308 8  8 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 308 9  8 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 308 10  10 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 308 11  10 3 15 1 1 1 1 1 1 0.000125 > Run_info/SYS_308_RUN_11.txt &
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
