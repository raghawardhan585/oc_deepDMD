#!/bin/bash
python3 gen_control_Koopman_SHARA_addition.py '/gpu:0' True &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:1' False &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:2' False &
python3 gen_control_Koopman_SHARA_addition.py '/gpu:3' False &

echo "All sessions are complete"