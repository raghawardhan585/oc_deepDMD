#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 0  2 3 5 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 1  2 3 5 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 2  2 3 5 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 3  2 3 5 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 4  2 3 10 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 5  2 3 10 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 6  2 3 10 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 7  2 3 10 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 8  2 3 15 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 9  2 3 15 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 10  2 3 15 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 11  2 3 15 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 12  2 4 5 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 13  2 4 5 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 14  2 4 5 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 15  2 4 5 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 16  2 4 10 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 17  2 4 10 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 18  2 4 10 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 19  2 4 10 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 20  2 4 15 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 21  2 4 15 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 22  2 4 15 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 23  2 4 15 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 24  2 5 5 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 25  2 5 5 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 26  2 5 5 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 27  2 5 5 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 28  2 5 10 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 29  2 5 10 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 30  2 5 10 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 31  2 5 10 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_31.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 32  2 5 15 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_32.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 33  2 5 15 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_33.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 34  2 5 15 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_34.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 35  2 5 15 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_35.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 36  2 6 5 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_36.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 37  2 6 5 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_37.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 38  2 6 5 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_38.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 39  2 6 5 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_39.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 40  2 6 10 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_40.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 41  2 6 10 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_41.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 42  2 6 10 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_42.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 43  2 6 10 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_43.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 402 44  2 6 15 1 1 1 1 1 1 0.01 > Run_info/SYS_402_RUN_44.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 402 45  2 6 15 1 1 1 1 1 1 0.05 > Run_info/SYS_402_RUN_45.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 402 46  2 6 15 1 1 1 1 1 1 0.1 > Run_info/SYS_402_RUN_46.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 402 47  2 6 15 1 1 1 1 1 1 0.5 > Run_info/SYS_402_RUN_47.txt &
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
