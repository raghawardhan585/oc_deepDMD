#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 0  3 6 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 1  3 6 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 2  3 9 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 3  3 9 12 1 1 1 1 1 1 > Run_info/SYS_53_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 4  4 3 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 5  4 3 12 1 1 1 1 1 1 > Run_info/SYS_53_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 6  4 3 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 7  4 3 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 8  4 9 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 9  4 9 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 10  4 12 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 11  4 12 12 1 1 1 1 1 1 > Run_info/SYS_53_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 12  5 6 9 1 1 1 1 1 1 > Run_info/SYS_53_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 13  5 6 12 1 1 1 1 1 1 > Run_info/SYS_53_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 53 14  5 6 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 53 15  5 6 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 53 16  5 12 15 1 1 1 1 1 1 > Run_info/SYS_53_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 53 17  5 12 18 1 1 1 1 1 1 > Run_info/SYS_53_RUN_17.txt &
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
