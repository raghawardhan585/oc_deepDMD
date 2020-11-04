#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 1 0 1 0 0 1 3 10 > Run_info/SYS_1_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 1 0 1 0 1 1 3 10 > Run_info/SYS_1_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 1 0 1 0 2 1 3 10 > Run_info/SYS_1_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 1 0 1 0 3 2 3 10 > Run_info/SYS_1_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 1 0 1 0 4 2 3 10 > Run_info/SYS_1_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 1 0 1 0 5 2 3 10 > Run_info/SYS_1_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 1 0 1 0 6 3 3 10 > Run_info/SYS_1_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 1 0 1 0 7 3 3 10 > Run_info/SYS_1_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 1 0 1 0 8 3 3 10 > Run_info/SYS_1_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 1 0 1 0 9 4 3 10 > Run_info/SYS_1_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 1 0 1 0 10 4 3 10 > Run_info/SYS_1_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 1 0 1 0 11 4 3 10 > Run_info/SYS_1_RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 1 0 1 0 12 5 3 10 > Run_info/SYS_1_RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 1 0 1 0 13 5 3 10 > Run_info/SYS_1_RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 1 0 1 0 14 5 3 10 > Run_info/SYS_1_RUN_14.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
