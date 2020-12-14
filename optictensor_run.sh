#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 0  7 7 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 1  7 8 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 2  7 8 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 3  7 8 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 4  7 11 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 5  7 11 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 6  7 11 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_6.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 7  8 9 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_7.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 8  8 9 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_8.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 9  8 11 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 10  9 7 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_10.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 11  9 9 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_11.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 12  9 10 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 13  10 7 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 14  10 8 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 15  10 8 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_15.txt &
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 16  10 10 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 17  10 11 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 18  10 11 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 19  10 11 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_19.txt &
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
