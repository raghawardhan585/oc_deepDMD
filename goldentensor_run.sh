#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 3 0 1 0 0 5 3 5 > Run_info/SYS_3_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 3 0 1 0 1 5 3 5 > Run_info/SYS_3_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 3 0 1 0 2 5 3 5 > Run_info/SYS_3_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 3 0 1 0 3 5 3 10 > Run_info/SYS_3_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 3 0 1 0 4 5 3 10 > Run_info/SYS_3_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 3 0 1 0 5 5 3 10 > Run_info/SYS_3_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 3 0 1 0 6 5 3 15 > Run_info/SYS_3_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 3 0 1 0 7 5 3 15 > Run_info/SYS_3_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 3 0 1 0 8 5 3 15 > Run_info/SYS_3_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 3 0 1 0 9 5 4 5 > Run_info/SYS_3_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 3 0 1 0 10 5 4 5 > Run_info/SYS_3_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 3 0 1 0 11 5 4 5 > Run_info/SYS_3_RUN_11.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
