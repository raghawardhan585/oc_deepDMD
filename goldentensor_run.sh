#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 0  2 4 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 1  2 5 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 2  2 5 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 3  2 5 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 4  2 6 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 5  2 6 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 6  2 7 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 7  2 7 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 8  2 8 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 9  2 8 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 10  2 8 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 11  3 4 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 12  3 5 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 13  3 5 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 14  3 5 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 15  3 5 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 16  3 7 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 17  3 7 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 18  3 7 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 19  3 7 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 20  3 8 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 21  4 4 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 22  4 4 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 23  4 4 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 24  4 5 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 25  4 5 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 26  4 6 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 27  4 6 10 1 1 1 1 1 1 > Run_info/SYS_27_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 27 28  4 7 15 1 1 1 1 1 1 > Run_info/SYS_27_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 27 29  4 7 20 1 1 1 1 1 1 > Run_info/SYS_27_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 27 30  4 7 25 1 1 1 1 1 1 > Run_info/SYS_27_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 27 31  4 8 5 1 1 1 1 1 1 > Run_info/SYS_27_RUN_31.txt &
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
