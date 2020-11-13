#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 4 0 1 0 0 3 4 5 > Run_info/SYS_4_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 4 0 1 0 1 3 4 5 > Run_info/SYS_4_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 4 0 1 0 2 3 4 5 > Run_info/SYS_4_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 4 0 1 0 3 3 4 10 > Run_info/SYS_4_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 4 0 1 0 4 3 4 10 > Run_info/SYS_4_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 4 0 1 0 5 3 4 10 > Run_info/SYS_4_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 4 0 1 0 6 3 4 15 > Run_info/SYS_4_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 4 0 1 0 7 3 4 15 > Run_info/SYS_4_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 4 0 1 0 8 3 4 15 > Run_info/SYS_4_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 4 0 1 0 9 3 6 5 > Run_info/SYS_4_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 4 0 1 0 10 3 6 5 > Run_info/SYS_4_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 4 0 1 0 11 3 6 5 > Run_info/SYS_4_RUN_11.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
