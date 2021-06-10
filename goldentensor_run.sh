#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 1  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 2  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 3  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 4  1 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 5  1 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 6  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 7  1 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 8  2 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 9  2 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 10  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 11  2 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 12  3 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 13  3 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 14  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 15  3 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 16  4 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 17  4 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 18  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 19  4 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 20  5 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 21  5 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 22  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 23  5 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 24  6 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 25  6 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 26  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 27  6 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 500 28  7 3 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 500 29  7 3 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 500 30  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 500 31  7 4 15 1 1 1 1 1 1 0 > Run_info/SYS_500_RUN_31.txt &
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
