#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 1  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 2  1 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 3  1 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 4  2 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 5  2 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 6  3 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 7  3 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 8  4 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 9  4 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 10  5 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 11  5 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 12  6 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 13  6 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 14  7 3 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 15  7 4 15 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_15.txt &
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
