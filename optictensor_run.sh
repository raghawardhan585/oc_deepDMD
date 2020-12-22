#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 0  9 3 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 1  9 3 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 2  9 3 21 1 1 1 1 1 1 > Run_info/SYS_53_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 3  9 6 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 4  10 3 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 5  10 3 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 6  10 3 21 1 1 1 1 1 1 > Run_info/SYS_53_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 7  10 6 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 8  11 3 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 9  11 3 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 10  11 3 21 1 1 1 1 1 1 > Run_info/SYS_53_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 11  11 6 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_11.txt &
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
