#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 0  10 3 15 1 1 1 1 1 1 0.0 > Run_info/SYS_400_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 1  10 3 15 1 1 1 1 1 1 2e-05 > Run_info/SYS_400_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 2  10 3 15 1 1 1 1 1 1 2.5e-05 > Run_info/SYS_400_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 3  10 3 15 1 1 1 1 1 1 3e-05 > Run_info/SYS_400_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 4  10 3 15 1 1 1 1 1 1 3.5e-05 > Run_info/SYS_400_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 5  10 3 15 1 1 1 1 1 1 3.9999999999999996e-05 > Run_info/SYS_400_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 6  10 3 15 1 1 1 1 1 1 4.4999999999999996e-05 > Run_info/SYS_400_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 7  10 3 15 1 1 1 1 1 1 4.9999999999999996e-05 > Run_info/SYS_400_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 8  10 3 15 1 1 1 1 1 1 5.4999999999999995e-05 > Run_info/SYS_400_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 9  10 3 15 1 1 1 1 1 1 5.9999999999999995e-05 > Run_info/SYS_400_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 10  10 3 15 1 1 1 1 1 1 6.5e-05 > Run_info/SYS_400_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 11  10 3 15 1 1 1 1 1 1 7e-05 > Run_info/SYS_400_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 12  10 3 15 1 1 1 1 1 1 7.5e-05 > Run_info/SYS_400_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 13  10 3 15 1 1 1 1 1 1 7.999999999999999e-05 > Run_info/SYS_400_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 14  10 3 15 1 1 1 1 1 1 8.499999999999999e-05 > Run_info/SYS_400_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 15  10 3 15 1 1 1 1 1 1 8.999999999999999e-05 > Run_info/SYS_400_RUN_15.txt &
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
