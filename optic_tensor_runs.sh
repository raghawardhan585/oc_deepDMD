#!/bin/bash
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' True > Run_info/Run_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' False > Run_info/Run_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' False > Run_info/Run_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' False > Run_info/Run_3.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' False > Run_info/Run_4.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' False > Run_info/Run_5.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' False > Run_info/Run_6.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' False > Run_info/Run_7.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' False > Run_info/Run_8.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' False > Run_info/Run_9.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' False > Run_info/Run_10.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' False > Run_info/Run_11.txt &
Echo "Running all sessions"
wait
echo "All sessions are complete"