#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 0 4 3 9 > Run_info/SYS_2_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 1 4 3 9 > Run_info/SYS_2_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 2 4 3 9 > Run_info/SYS_2_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 3 4 3 9 > Run_info/SYS_2_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 4 4 3 9 > Run_info/SYS_2_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 5 4 3 12 > Run_info/SYS_2_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 6 4 3 12 > Run_info/SYS_2_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 7 4 3 12 > Run_info/SYS_2_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 8 4 3 12 > Run_info/SYS_2_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 9 4 3 12 > Run_info/SYS_2_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 10 4 3 15 > Run_info/SYS_2_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 11 4 3 15 > Run_info/SYS_2_RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 12 4 3 15 > Run_info/SYS_2_RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 13 4 3 15 > Run_info/SYS_2_RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 14 4 3 15 > Run_info/SYS_2_RUN_14.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 15 4 3 18 > Run_info/SYS_2_RUN_15.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 16 4 3 18 > Run_info/SYS_2_RUN_16.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 17 4 3 18 > Run_info/SYS_2_RUN_17.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 18 4 3 18 > Run_info/SYS_2_RUN_18.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 19 4 3 18 > Run_info/SYS_2_RUN_19.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
