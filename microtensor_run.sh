#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [with_u] [with_y] [mix_xu] [run_no] [dict_size] [nn_layers] [nn_nodes] [write_to_file] 
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 4 0 1 0 0 4 7 15 > Run_info/SYS_4_RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 4 0 1 0 1 4 7 15 > Run_info/SYS_4_RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 4 0 1 0 2 4 7 15 > Run_info/SYS_4_RUN_2.txt &
echo "Running all sessions" 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
