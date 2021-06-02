#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 0  1 1 1 0 1 0 1 1 1 0 > Run_info/SYS_400_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 1  1 1 1 0 1 0 1 1 1 0 > Run_info/SYS_400_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 2  1 1 1 0 1 0 1 1 1 0 > Run_info/SYS_400_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 3  1 1 1 0 1 0 1 1 1 0 > Run_info/SYS_400_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 4  1 1 1 1 3 5 1 1 1 0 > Run_info/SYS_400_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 5  1 1 1 1 3 10 1 1 1 0 > Run_info/SYS_400_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 6  1 1 1 1 4 5 1 1 1 0 > Run_info/SYS_400_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 7  1 1 1 1 4 10 1 1 1 0 > Run_info/SYS_400_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 8  1 1 1 2 3 5 1 1 1 0 > Run_info/SYS_400_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 9  1 1 1 2 3 10 1 1 1 0 > Run_info/SYS_400_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 10  1 1 1 2 4 5 1 1 1 0 > Run_info/SYS_400_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 11  1 1 1 2 4 10 1 1 1 0 > Run_info/SYS_400_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 12  1 1 1 3 3 5 1 1 1 0 > Run_info/SYS_400_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 13  1 1 1 3 3 10 1 1 1 0 > Run_info/SYS_400_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 14  1 1 1 3 4 5 1 1 1 0 > Run_info/SYS_400_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 15  1 1 1 3 4 10 1 1 1 0 > Run_info/SYS_400_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 16  1 1 1 4 3 5 1 1 1 0 > Run_info/SYS_400_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 17  1 1 1 4 3 10 1 1 1 0 > Run_info/SYS_400_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 18  1 1 1 4 4 5 1 1 1 0 > Run_info/SYS_400_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 19  1 1 1 4 4 10 1 1 1 0 > Run_info/SYS_400_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 20  1 1 1 5 3 5 1 1 1 0 > Run_info/SYS_400_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 21  1 1 1 5 3 10 1 1 1 0 > Run_info/SYS_400_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 22  1 1 1 5 4 5 1 1 1 0 > Run_info/SYS_400_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 23  1 1 1 5 4 10 1 1 1 0 > Run_info/SYS_400_RUN_23.txt &
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
