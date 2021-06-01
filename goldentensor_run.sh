#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 0  4 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 1  4 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 2  4 5 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 3  4 6 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 4  4 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 5  4 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 6  4 5 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 7  4 6 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 8  8 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 9  8 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 10  8 5 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 11  8 6 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 12  8 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 13  8 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 14  8 5 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 15  8 6 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 16  10 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 17  10 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 18  10 5 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 19  10 6 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 20  10 3 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 21  10 4 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 22  10 5 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 23  10 6 10 1 1 1 1 1 1 6e-05 > Run_info/SYS_400_RUN_23.txt &
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
