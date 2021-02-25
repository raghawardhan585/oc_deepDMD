#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 0  1 3 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 1  1 3 5 1 1 1 1 1 1 > Run_info/SYS_11_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 2  1 4 2 1 1 1 1 1 1 > Run_info/SYS_11_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 3  1 4 3 1 1 1 1 1 1 > Run_info/SYS_11_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 4  1 8 2 1 1 1 1 1 1 > Run_info/SYS_11_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 5  1 8 3 1 1 1 1 1 1 > Run_info/SYS_11_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 6  1 8 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 7  1 8 5 1 1 1 1 1 1 > Run_info/SYS_11_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 8  2 3 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 9  2 3 5 1 1 1 1 1 1 > Run_info/SYS_11_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 10  2 4 2 1 1 1 1 1 1 > Run_info/SYS_11_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 11  2 4 3 1 1 1 1 1 1 > Run_info/SYS_11_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 12  2 8 2 1 1 1 1 1 1 > Run_info/SYS_11_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 13  2 8 3 1 1 1 1 1 1 > Run_info/SYS_11_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 14  2 8 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 15  2 8 5 1 1 1 1 1 1 > Run_info/SYS_11_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 16  3 3 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 17  3 3 5 1 1 1 1 1 1 > Run_info/SYS_11_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 18  3 4 2 1 1 1 1 1 1 > Run_info/SYS_11_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 19  3 4 3 1 1 1 1 1 1 > Run_info/SYS_11_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 11 20  3 8 2 1 1 1 1 1 1 > Run_info/SYS_11_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 11 21  3 8 3 1 1 1 1 1 1 > Run_info/SYS_11_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 11 22  3 8 4 1 1 1 1 1 1 > Run_info/SYS_11_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 11 23  3 8 5 1 1 1 1 1 1 > Run_info/SYS_11_RUN_23.txt &
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
