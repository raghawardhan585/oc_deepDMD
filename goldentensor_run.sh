#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [n_observables] [n_layers] [n_nodes] [write_to_file] 
python3 deepDMD.py '/gpu:0' 416 0 0 3 10 > Run_info/SYS_416_RUN_0.txt &
python3 deepDMD.py '/gpu:1' 416 1 0 4 10 > Run_info/SYS_416_RUN_1.txt &
python3 deepDMD.py '/gpu:2' 416 2 0 5 10 > Run_info/SYS_416_RUN_2.txt &
python3 deepDMD.py '/gpu:3' 416 3 1 3 10 > Run_info/SYS_416_RUN_3.txt &
wait 
python3 deepDMD.py '/gpu:0' 416 4 1 4 10 > Run_info/SYS_416_RUN_4.txt &
python3 deepDMD.py '/gpu:1' 416 5 1 5 10 > Run_info/SYS_416_RUN_5.txt &
python3 deepDMD.py '/gpu:2' 416 6 2 3 10 > Run_info/SYS_416_RUN_6.txt &
python3 deepDMD.py '/gpu:3' 416 7 2 4 10 > Run_info/SYS_416_RUN_7.txt &
wait 
python3 deepDMD.py '/gpu:0' 416 8 2 5 10 > Run_info/SYS_416_RUN_8.txt &
python3 deepDMD.py '/gpu:1' 416 9 3 3 10 > Run_info/SYS_416_RUN_9.txt &
python3 deepDMD.py '/gpu:2' 416 10 3 4 10 > Run_info/SYS_416_RUN_10.txt &
python3 deepDMD.py '/gpu:3' 416 11 3 5 10 > Run_info/SYS_416_RUN_11.txt &
wait 
python3 deepDMD.py '/gpu:0' 416 12 4 3 10 > Run_info/SYS_416_RUN_12.txt &
python3 deepDMD.py '/gpu:1' 416 13 4 4 10 > Run_info/SYS_416_RUN_13.txt &
python3 deepDMD.py '/gpu:2' 416 14 4 5 10 > Run_info/SYS_416_RUN_14.txt &
python3 deepDMD.py '/gpu:3' 416 15 5 3 10 > Run_info/SYS_416_RUN_15.txt &
wait 
python3 deepDMD.py '/gpu:0' 416 16 5 4 10 > Run_info/SYS_416_RUN_16.txt &
python3 deepDMD.py '/gpu:1' 416 17 5 5 10 > Run_info/SYS_416_RUN_17.txt &
python3 deepDMD.py '/gpu:2' 416 18 6 3 10 > Run_info/SYS_416_RUN_18.txt &
python3 deepDMD.py '/gpu:3' 416 19 6 4 10 > Run_info/SYS_416_RUN_19.txt &
wait 
python3 deepDMD.py '/gpu:0' 416 20 6 5 10 > Run_info/SYS_416_RUN_20.txt &
python3 deepDMD.py '/gpu:1' 416 21 7 3 10 > Run_info/SYS_416_RUN_21.txt &
python3 deepDMD.py '/gpu:2' 416 22 7 4 10 > Run_info/SYS_416_RUN_22.txt &
python3 deepDMD.py '/gpu:3' 416 23 7 5 10 > Run_info/SYS_416_RUN_23.txt &
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
