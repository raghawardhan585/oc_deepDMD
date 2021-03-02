#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 0  1 1 1 1 3 6 1 1 1 > Run_info/SYS_91_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 1  1 1 1 1 3 8 1 1 1 > Run_info/SYS_91_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 2  1 1 1 1 4 2 1 1 1 > Run_info/SYS_91_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 3  1 1 1 1 4 4 1 1 1 > Run_info/SYS_91_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 4  1 1 1 1 9 2 1 1 1 > Run_info/SYS_91_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 5  1 1 1 1 9 4 1 1 1 > Run_info/SYS_91_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 6  1 1 1 1 9 6 1 1 1 > Run_info/SYS_91_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 7  1 1 1 1 9 8 1 1 1 > Run_info/SYS_91_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 91 8  1 1 1 1 4 6 1 1 1 > Run_info/SYS_91_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 91 9  1 1 1 1 4 8 1 1 1 > Run_info/SYS_91_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 91 10  1 1 1 1 8 2 1 1 1 > Run_info/SYS_91_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 91 11  1 1 1 1 8 4 1 1 1 > Run_info/SYS_91_RUN_11.txt &
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
