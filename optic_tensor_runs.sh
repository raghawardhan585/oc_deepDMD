#!/bin/bash
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 0 > Run_info/RUN_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 1 > Run_info/RUN_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 2 > Run_info/RUN_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 3 > Run_info/RUN_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 4 > Run_info/RUN_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 5 > Run_info/RUN_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 6 > Run_info/RUN_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 7 > Run_info/RUN_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' 8 > Run_info/RUN_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' 9 > Run_info/RUN_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' 10 > Run_info/RUN_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' 11 > Run_info/RUN_11.txt &
python3 gen_control_Koopman_SHARA_addition.py '/cpu:0' 12 > Run_info/RUN_12.txt &
echo "Running all sessions"
wait
echo "All sessions are complete"