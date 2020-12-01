#!/bin/bash 
rm nohup.out
git fetch --all
git reset --hard origin
cd ..
cp -a _current_run_saved_files/ oc_deepDMD/_current_run_saved_files
cp -a Run_info/ oc_deepDMD/Run_info
cd oc_deepDMD
git add .
git commit -m "MT runs complete"
git pull
git push