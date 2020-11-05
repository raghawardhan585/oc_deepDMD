#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 0 2 3 6 > Run_info/SYS_2_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 1 2 3 6 > Run_info/SYS_2_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 2 2 3 6 > Run_info/SYS_2_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 3 3 3 6 > Run_info/SYS_2_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 4 3 3 6 > Run_info/SYS_2_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 5 3 3 6 > Run_info/SYS_2_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 6 4 3 6 > Run_info/SYS_2_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 7 4 3 6 > Run_info/SYS_2_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 8 4 3 6 > Run_info/SYS_2_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 9 5 3 6 > Run_info/SYS_2_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 10 5 3 6 > Run_info/SYS_2_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 11 5 3 6 > Run_info/SYS_2_RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 12 6 3 6 > Run_info/SYS_2_RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 13 6 3 6 > Run_info/SYS_2_RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 14 6 3 6 > Run_info/SYS_2_RUN_14.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
