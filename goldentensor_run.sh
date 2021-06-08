#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 0  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 1  0 1 0 1 1 1 1 1 1 5e-06 > Run_info/SYS_406_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 2  0 1 0 1 1 1 1 1 1 1e-05 > Run_info/SYS_406_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 3  0 1 0 1 1 1 1 1 1 5e-05 > Run_info/SYS_406_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 4  0 1 0 1 1 1 1 1 1 0.0001 > Run_info/SYS_406_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 5  0 1 0 1 1 1 1 1 1 0.0005 > Run_info/SYS_406_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 6  0 1 0 1 1 1 1 1 1 0.001 > Run_info/SYS_406_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 7  0 1 0 1 1 1 1 1 1 0.005 > Run_info/SYS_406_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 8  0 1 0 1 1 1 1 1 1 0.01 > Run_info/SYS_406_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 9  0 1 0 1 1 1 1 1 1 0.05 > Run_info/SYS_406_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 10  0 1 0 1 1 1 1 1 1 0.1 > Run_info/SYS_406_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 11  0 1 0 1 1 1 1 1 1 0.5 > Run_info/SYS_406_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 12  0 1 0 1 1 1 1 1 1 0 > Run_info/SYS_406_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 13  0 1 0 1 1 1 1 1 1 5e-06 > Run_info/SYS_406_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 14  0 1 0 1 1 1 1 1 1 1e-05 > Run_info/SYS_406_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 15  0 1 0 1 1 1 1 1 1 5e-05 > Run_info/SYS_406_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 16  0 1 0 1 1 1 1 1 1 0.0001 > Run_info/SYS_406_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 17  0 1 0 1 1 1 1 1 1 0.0005 > Run_info/SYS_406_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 18  0 1 0 1 1 1 1 1 1 0.001 > Run_info/SYS_406_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 19  0 1 0 1 1 1 1 1 1 0.005 > Run_info/SYS_406_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 406 20  0 1 0 1 1 1 1 1 1 0.01 > Run_info/SYS_406_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 406 21  0 1 0 1 1 1 1 1 1 0.05 > Run_info/SYS_406_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 406 22  0 1 0 1 1 1 1 1 1 0.1 > Run_info/SYS_406_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 406 23  0 1 0 1 1 1 1 1 1 0.5 > Run_info/SYS_406_RUN_23.txt &
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
