#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 0 4 4 6 > Run_info/SYS_2_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 1 4 4 6 > Run_info/SYS_2_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 2 4 4 6 > Run_info/SYS_2_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 3 4 4 6 > Run_info/SYS_2_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 4 4 4 6 > Run_info/SYS_2_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 5 4 4 9 > Run_info/SYS_2_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 6 4 4 9 > Run_info/SYS_2_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 7 4 4 9 > Run_info/SYS_2_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 8 4 4 9 > Run_info/SYS_2_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 9 4 4 9 > Run_info/SYS_2_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 10 4 4 12 > Run_info/SYS_2_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 11 4 4 12 > Run_info/SYS_2_RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 12 4 4 12 > Run_info/SYS_2_RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 13 4 4 12 > Run_info/SYS_2_RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 14 4 4 12 > Run_info/SYS_2_RUN_14.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 15 4 4 15 > Run_info/SYS_2_RUN_15.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 16 4 4 15 > Run_info/SYS_2_RUN_16.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 17 4 4 15 > Run_info/SYS_2_RUN_17.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 18 4 4 15 > Run_info/SYS_2_RUN_18.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 19 4 4 15 > Run_info/SYS_2_RUN_19.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 20 4 4 18 > Run_info/SYS_2_RUN_20.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 21 4 4 18 > Run_info/SYS_2_RUN_21.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 22 4 4 18 > Run_info/SYS_2_RUN_22.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 23 4 4 18 > Run_info/SYS_2_RUN_23.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 24 4 4 18 > Run_info/SYS_2_RUN_24.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
