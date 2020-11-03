#!/bin/bash
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' True > RunGpu_0.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' False > RunGpu_1.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' False > RunGpu_2.txt &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' False > RunGpu_3.txt &

echo "All sessions are complete"