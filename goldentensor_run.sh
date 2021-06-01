#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 0  5 3 22 1 1 1 1 1 1 0.0 > Run_info/SYS_307_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 1  5 3 22 1 1 1 1 1 1 2.5e-05 > Run_info/SYS_307_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 2  5 3 22 1 1 1 1 1 1 5e-05 > Run_info/SYS_307_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 3  5 3 22 1 1 1 1 1 1 7.500000000000001e-05 > Run_info/SYS_307_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 4  5 3 22 1 1 1 1 1 1 0.0001 > Run_info/SYS_307_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 5  5 3 22 1 1 1 1 1 1 0.000125 > Run_info/SYS_307_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 6  5 3 22 1 1 1 1 1 1 0.00015000000000000001 > Run_info/SYS_307_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 7  5 3 22 1 1 1 1 1 1 0.000175 > Run_info/SYS_307_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 8  5 3 22 1 1 1 1 1 1 0.0002 > Run_info/SYS_307_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 9  5 3 22 1 1 1 1 1 1 0.00022500000000000002 > Run_info/SYS_307_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 10  5 3 22 1 1 1 1 1 1 0.00025 > Run_info/SYS_307_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 11  5 3 22 1 1 1 1 1 1 0.000275 > Run_info/SYS_307_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 12  5 3 22 1 1 1 1 1 1 0.00030000000000000003 > Run_info/SYS_307_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 13  5 3 22 1 1 1 1 1 1 0.00032500000000000004 > Run_info/SYS_307_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 14  5 3 22 1 1 1 1 1 1 0.00035 > Run_info/SYS_307_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 15  5 3 22 1 1 1 1 1 1 0.000375 > Run_info/SYS_307_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 16  5 3 22 1 1 1 1 1 1 0.0004 > Run_info/SYS_307_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 17  5 3 22 1 1 1 1 1 1 0.00042500000000000003 > Run_info/SYS_307_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 18  5 3 22 1 1 1 1 1 1 0.00045000000000000004 > Run_info/SYS_307_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 19  5 3 22 1 1 1 1 1 1 0.000475 > Run_info/SYS_307_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 20  5 3 22 1 1 1 1 1 1 0.0005 > Run_info/SYS_307_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 21  5 3 22 1 1 1 1 1 1 0.0005250000000000001 > Run_info/SYS_307_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 22  5 3 22 1 1 1 1 1 1 0.00055 > Run_info/SYS_307_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 23  5 3 22 1 1 1 1 1 1 0.000575 > Run_info/SYS_307_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 24  5 3 22 1 1 1 1 1 1 0.0006000000000000001 > Run_info/SYS_307_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 25  5 3 22 1 1 1 1 1 1 0.000625 > Run_info/SYS_307_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 26  5 3 22 1 1 1 1 1 1 0.0006500000000000001 > Run_info/SYS_307_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 27  5 3 22 1 1 1 1 1 1 0.000675 > Run_info/SYS_307_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 28  5 3 22 1 1 1 1 1 1 0.0007 > Run_info/SYS_307_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 29  5 3 22 1 1 1 1 1 1 0.0007250000000000001 > Run_info/SYS_307_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 30  5 3 22 1 1 1 1 1 1 0.00075 > Run_info/SYS_307_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 31  5 3 22 1 1 1 1 1 1 0.0007750000000000001 > Run_info/SYS_307_RUN_31.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 32  5 3 22 1 1 1 1 1 1 0.0008 > Run_info/SYS_307_RUN_32.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 33  5 3 22 1 1 1 1 1 1 0.000825 > Run_info/SYS_307_RUN_33.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 34  5 3 22 1 1 1 1 1 1 0.0008500000000000001 > Run_info/SYS_307_RUN_34.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 35  5 3 22 1 1 1 1 1 1 0.000875 > Run_info/SYS_307_RUN_35.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 307 36  5 3 22 1 1 1 1 1 1 0.0009000000000000001 > Run_info/SYS_307_RUN_36.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 307 37  5 3 22 1 1 1 1 1 1 0.000925 > Run_info/SYS_307_RUN_37.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 307 38  5 3 22 1 1 1 1 1 1 0.00095 > Run_info/SYS_307_RUN_38.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 307 39  5 3 22 1 1 1 1 1 1 0.0009750000000000001 > Run_info/SYS_307_RUN_39.txt &
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
