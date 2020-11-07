#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 0 1 4 12 > Run_info/SYS_2_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 1 1 4 12 > Run_info/SYS_2_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 2 1 4 12 > Run_info/SYS_2_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 3 1 4 12 > Run_info/SYS_2_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 2 0 1 0 4 1 4 12 > Run_info/SYS_2_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 5 1 4 15 > Run_info/SYS_2_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 6 1 4 15 > Run_info/SYS_2_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 7 1 4 15 > Run_info/SYS_2_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 8 1 4 15 > Run_info/SYS_2_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 2 0 1 0 9 1 4 15 > Run_info/SYS_2_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 10 1 4 18 > Run_info/SYS_2_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 11 1 4 18 > Run_info/SYS_2_RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 12 1 4 18 > Run_info/SYS_2_RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 13 1 4 18 > Run_info/SYS_2_RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 0 1 0 14 1 4 18 > Run_info/SYS_2_RUN_14.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 15 2 4 6 > Run_info/SYS_2_RUN_15.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 16 2 4 6 > Run_info/SYS_2_RUN_16.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 17 2 4 6 > Run_info/SYS_2_RUN_17.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 18 2 4 6 > Run_info/SYS_2_RUN_18.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 2 0 1 0 19 2 4 6 > Run_info/SYS_2_RUN_19.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
