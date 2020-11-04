#!/bin/bash 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py 'gpu:0' 0 5 3 10 > Run_info/RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:0' 1 5 3 10 > Run_info/RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:0' 2 5 3 10 > Run_info/RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:1' 3 4 3 10 > Run_info/RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:1' 4 4 3 10 > Run_info/RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:1' 5 4 3 10 > Run_info/RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:2' 6 4 3 10 > Run_info/RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:2' 7 4 3 10 > Run_info/RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:2' 8 4 3 10 > Run_info/RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:3' 9 4 3 10 > Run_info/RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:3' 10 4 3 10 > Run_info/RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py 'gpu:3' 11 4 3 10 > Run_info/RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py 'cpu:0' 12 4 3 10 > Run_info/RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py 'cpu:0' 13 4 3 10 > Run_info/RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py 'cpu:0' 14 4 3 10 > Run_info/RUN_14.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
