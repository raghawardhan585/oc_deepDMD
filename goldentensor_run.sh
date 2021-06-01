#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 0  0 1 0 1 1 1 1 1 1 0.0 > Run_info/SYS_400_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 1  0 1 0 1 1 1 1 1 1 2.5e-06 > Run_info/SYS_400_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 2  0 1 0 1 1 1 1 1 1 5e-06 > Run_info/SYS_400_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 3  0 1 0 1 1 1 1 1 1 7.500000000000001e-06 > Run_info/SYS_400_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 4  0 1 0 1 1 1 1 1 1 1e-05 > Run_info/SYS_400_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 5  0 1 0 1 1 1 1 1 1 1.25e-05 > Run_info/SYS_400_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 6  0 1 0 1 1 1 1 1 1 1.5000000000000002e-05 > Run_info/SYS_400_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 7  0 1 0 1 1 1 1 1 1 1.7500000000000002e-05 > Run_info/SYS_400_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 8  0 1 0 1 1 1 1 1 1 2e-05 > Run_info/SYS_400_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 9  0 1 0 1 1 1 1 1 1 2.25e-05 > Run_info/SYS_400_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 10  0 1 0 1 1 1 1 1 1 2.5e-05 > Run_info/SYS_400_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 11  0 1 0 1 1 1 1 1 1 2.75e-05 > Run_info/SYS_400_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 12  0 1 0 1 1 1 1 1 1 3.0000000000000004e-05 > Run_info/SYS_400_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 13  0 1 0 1 1 1 1 1 1 3.2500000000000004e-05 > Run_info/SYS_400_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 14  0 1 0 1 1 1 1 1 1 3.5000000000000004e-05 > Run_info/SYS_400_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 15  0 1 0 1 1 1 1 1 1 3.7500000000000003e-05 > Run_info/SYS_400_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 16  0 1 0 1 1 1 1 1 1 4e-05 > Run_info/SYS_400_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 17  0 1 0 1 1 1 1 1 1 4.25e-05 > Run_info/SYS_400_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 18  0 1 0 1 1 1 1 1 1 4.5e-05 > Run_info/SYS_400_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 19  0 1 0 1 1 1 1 1 1 4.75e-05 > Run_info/SYS_400_RUN_19.txt &
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
