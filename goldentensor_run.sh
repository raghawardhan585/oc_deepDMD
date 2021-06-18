#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 1  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 2  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 3  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 4  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 5  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 6  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 7  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 8  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 9  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 10  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 11  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 12  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 13  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 14  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 15  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 16  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 17  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 18  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 19  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 20  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 21  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 22  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 23  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 24  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 25  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 26  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 27  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 701 28  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 701 29  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 701 30  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 701 31  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_701_RUN_31.txt &
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
