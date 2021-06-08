#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 0  4 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 1  4 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 2  4 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 3  4 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 4  4 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 5  4 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 6  4 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 7  4 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 8  5 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 9  5 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 10  5 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 11  5 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 12  5 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 13  5 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 14  5 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 15  5 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 16  6 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 17  6 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 18  6 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 19  6 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 20  6 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 21  6 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 22  6 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 23  6 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 24  7 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 25  7 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 26  7 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 27  7 3 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 28  7 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 29  7 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 30  7 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 31  7 4 20 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_31.txt &
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
