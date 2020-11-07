#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 0 3 5 6 > Run_info/SYS_2_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 1 3 5 6 > Run_info/SYS_2_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 2 3 5 6 > Run_info/SYS_2_RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 3 3 5 6 > Run_info/SYS_2_RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 4 3 5 6 > Run_info/SYS_2_RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 5 3 5 9 > Run_info/SYS_2_RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 6 3 5 9 > Run_info/SYS_2_RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 7 3 5 9 > Run_info/SYS_2_RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 8 3 5 9 > Run_info/SYS_2_RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 9 3 5 9 > Run_info/SYS_2_RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 10 3 5 12 > Run_info/SYS_2_RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 11 3 5 12 > Run_info/SYS_2_RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 12 3 5 12 > Run_info/SYS_2_RUN_12.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 13 3 5 12 > Run_info/SYS_2_RUN_13.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 14 3 5 12 > Run_info/SYS_2_RUN_14.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 15 3 5 15 > Run_info/SYS_2_RUN_15.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 16 3 5 15 > Run_info/SYS_2_RUN_16.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 17 3 5 15 > Run_info/SYS_2_RUN_17.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 18 3 5 15 > Run_info/SYS_2_RUN_18.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 19 3 5 15 > Run_info/SYS_2_RUN_19.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 20 3 5 18 > Run_info/SYS_2_RUN_20.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 21 3 5 18 > Run_info/SYS_2_RUN_21.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 22 3 5 18 > Run_info/SYS_2_RUN_22.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 23 3 5 18 > Run_info/SYS_2_RUN_23.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 24 3 5 18 > Run_info/SYS_2_RUN_24.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 25 4 5 6 > Run_info/SYS_2_RUN_25.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 26 4 5 6 > Run_info/SYS_2_RUN_26.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 27 4 5 6 > Run_info/SYS_2_RUN_27.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 28 4 5 6 > Run_info/SYS_2_RUN_28.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 29 4 5 6 > Run_info/SYS_2_RUN_29.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 30 4 5 9 > Run_info/SYS_2_RUN_30.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 31 4 5 9 > Run_info/SYS_2_RUN_31.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 32 4 5 9 > Run_info/SYS_2_RUN_32.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 33 4 5 9 > Run_info/SYS_2_RUN_33.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 2 0 1 0 34 4 5 9 > Run_info/SYS_2_RUN_34.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
