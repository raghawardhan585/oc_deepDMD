#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 704 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 704 1  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 704 2  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 704 3  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 704 4  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 704 5  1 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 704 6  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 704 7  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 704 8  2 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 704 9  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 704 10  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 704 11  3 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 704 12  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 704 13  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 704 14  4 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 704 15  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 704 16  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 704 17  5 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 704 18  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 704 19  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 704 20  6 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 704 21  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 704 22  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 704 23  7 4 10 1 1 1 1 1 1 0 > Run_info/SYS_704_RUN_23.txt &
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
