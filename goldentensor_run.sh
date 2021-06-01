#!/bin/bash 
rm -rf _current_run_saved_files 
mkdir _current_run_saved_files 
rm -rf Run_info 
mkdir Run_info 
# Gen syntax: [interpreter] [code.py] [device] [sys_no] [run_no] [x_dict] [x_layers] [x_nodes] [y_dict] [y_layers] [y_nodes] [xy_dict] [xy_layers] [xy_nodes] [regularization lambda] [write_to_file] 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 0  0 1 0 1 1 1 1 1 1 0.0 > Run_info/SYS_400_RUN_0.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 1  0 1 0 1 1 1 1 1 1 5e-05 > Run_info/SYS_400_RUN_1.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 2  0 1 0 1 1 1 1 1 1 0.0001 > Run_info/SYS_400_RUN_2.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 3  0 1 0 1 1 1 1 1 1 0.00015000000000000001 > Run_info/SYS_400_RUN_3.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 4  0 1 0 1 1 1 1 1 1 0.0002 > Run_info/SYS_400_RUN_4.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 5  0 1 0 1 1 1 1 1 1 0.00025 > Run_info/SYS_400_RUN_5.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 6  0 1 0 1 1 1 1 1 1 0.00030000000000000003 > Run_info/SYS_400_RUN_6.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 7  0 1 0 1 1 1 1 1 1 0.00035 > Run_info/SYS_400_RUN_7.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 8  0 1 0 1 1 1 1 1 1 0.0004 > Run_info/SYS_400_RUN_8.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 9  0 1 0 1 1 1 1 1 1 0.00045000000000000004 > Run_info/SYS_400_RUN_9.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 10  0 1 0 1 1 1 1 1 1 0.0005 > Run_info/SYS_400_RUN_10.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 11  0 1 0 1 1 1 1 1 1 0.00055 > Run_info/SYS_400_RUN_11.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 12  0 1 0 1 1 1 1 1 1 0.0006000000000000001 > Run_info/SYS_400_RUN_12.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 13  0 1 0 1 1 1 1 1 1 0.0006500000000000001 > Run_info/SYS_400_RUN_13.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 14  0 1 0 1 1 1 1 1 1 0.0007 > Run_info/SYS_400_RUN_14.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 15  0 1 0 1 1 1 1 1 1 0.00075 > Run_info/SYS_400_RUN_15.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 16  0 1 0 1 1 1 1 1 1 0.0008 > Run_info/SYS_400_RUN_16.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 17  0 1 0 1 1 1 1 1 1 0.0008500000000000001 > Run_info/SYS_400_RUN_17.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 18  0 1 0 1 1 1 1 1 1 0.0009000000000000001 > Run_info/SYS_400_RUN_18.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 19  0 1 0 1 1 1 1 1 1 0.00095 > Run_info/SYS_400_RUN_19.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 20  0 1 0 1 1 1 1 1 1 0.001 > Run_info/SYS_400_RUN_20.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 21  0 1 0 1 1 1 1 1 1 0.0010500000000000002 > Run_info/SYS_400_RUN_21.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 22  0 1 0 1 1 1 1 1 1 0.0011 > Run_info/SYS_400_RUN_22.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 23  0 1 0 1 1 1 1 1 1 0.00115 > Run_info/SYS_400_RUN_23.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 24  0 1 0 1 1 1 1 1 1 0.0012000000000000001 > Run_info/SYS_400_RUN_24.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 25  0 1 0 1 1 1 1 1 1 0.00125 > Run_info/SYS_400_RUN_25.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 26  0 1 0 1 1 1 1 1 1 0.0013000000000000002 > Run_info/SYS_400_RUN_26.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 27  0 1 0 1 1 1 1 1 1 0.00135 > Run_info/SYS_400_RUN_27.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 28  0 1 0 1 1 1 1 1 1 0.0014 > Run_info/SYS_400_RUN_28.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 29  0 1 0 1 1 1 1 1 1 0.0014500000000000001 > Run_info/SYS_400_RUN_29.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 30  0 1 0 1 1 1 1 1 1 0.0015 > Run_info/SYS_400_RUN_30.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 31  0 1 0 1 1 1 1 1 1 0.0015500000000000002 > Run_info/SYS_400_RUN_31.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 32  0 1 0 1 1 1 1 1 1 0.0016 > Run_info/SYS_400_RUN_32.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 33  0 1 0 1 1 1 1 1 1 0.00165 > Run_info/SYS_400_RUN_33.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 34  0 1 0 1 1 1 1 1 1 0.0017000000000000001 > Run_info/SYS_400_RUN_34.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 35  0 1 0 1 1 1 1 1 1 0.00175 > Run_info/SYS_400_RUN_35.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 36  0 1 0 1 1 1 1 1 1 0.0018000000000000002 > Run_info/SYS_400_RUN_36.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 37  0 1 0 1 1 1 1 1 1 0.00185 > Run_info/SYS_400_RUN_37.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 38  0 1 0 1 1 1 1 1 1 0.0019 > Run_info/SYS_400_RUN_38.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 39  0 1 0 1 1 1 1 1 1 0.0019500000000000001 > Run_info/SYS_400_RUN_39.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 40  0 1 0 1 1 1 1 1 1 0.002 > Run_info/SYS_400_RUN_40.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 41  0 1 0 1 1 1 1 1 1 0.00205 > Run_info/SYS_400_RUN_41.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 42  0 1 0 1 1 1 1 1 1 0.0021000000000000003 > Run_info/SYS_400_RUN_42.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 43  0 1 0 1 1 1 1 1 1 0.00215 > Run_info/SYS_400_RUN_43.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 44  0 1 0 1 1 1 1 1 1 0.0022 > Run_info/SYS_400_RUN_44.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 45  0 1 0 1 1 1 1 1 1 0.0022500000000000003 > Run_info/SYS_400_RUN_45.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 46  0 1 0 1 1 1 1 1 1 0.0023 > Run_info/SYS_400_RUN_46.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 47  0 1 0 1 1 1 1 1 1 0.00235 > Run_info/SYS_400_RUN_47.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 48  0 1 0 1 1 1 1 1 1 0.0024000000000000002 > Run_info/SYS_400_RUN_48.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 49  0 1 0 1 1 1 1 1 1 0.00245 > Run_info/SYS_400_RUN_49.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 50  0 1 0 1 1 1 1 1 1 0.0025 > Run_info/SYS_400_RUN_50.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 51  0 1 0 1 1 1 1 1 1 0.00255 > Run_info/SYS_400_RUN_51.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 52  0 1 0 1 1 1 1 1 1 0.0026000000000000003 > Run_info/SYS_400_RUN_52.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 53  0 1 0 1 1 1 1 1 1 0.00265 > Run_info/SYS_400_RUN_53.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 54  0 1 0 1 1 1 1 1 1 0.0027 > Run_info/SYS_400_RUN_54.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 55  0 1 0 1 1 1 1 1 1 0.0027500000000000003 > Run_info/SYS_400_RUN_55.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 56  0 1 0 1 1 1 1 1 1 0.0028 > Run_info/SYS_400_RUN_56.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 57  0 1 0 1 1 1 1 1 1 0.00285 > Run_info/SYS_400_RUN_57.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 58  0 1 0 1 1 1 1 1 1 0.0029000000000000002 > Run_info/SYS_400_RUN_58.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 59  0 1 0 1 1 1 1 1 1 0.00295 > Run_info/SYS_400_RUN_59.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 60  0 1 0 1 1 1 1 1 1 0.003 > Run_info/SYS_400_RUN_60.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 61  0 1 0 1 1 1 1 1 1 0.00305 > Run_info/SYS_400_RUN_61.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 62  0 1 0 1 1 1 1 1 1 0.0031000000000000003 > Run_info/SYS_400_RUN_62.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 63  0 1 0 1 1 1 1 1 1 0.00315 > Run_info/SYS_400_RUN_63.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 64  0 1 0 1 1 1 1 1 1 0.0032 > Run_info/SYS_400_RUN_64.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 65  0 1 0 1 1 1 1 1 1 0.0032500000000000003 > Run_info/SYS_400_RUN_65.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 66  0 1 0 1 1 1 1 1 1 0.0033 > Run_info/SYS_400_RUN_66.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 67  0 1 0 1 1 1 1 1 1 0.00335 > Run_info/SYS_400_RUN_67.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 68  0 1 0 1 1 1 1 1 1 0.0034000000000000002 > Run_info/SYS_400_RUN_68.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 69  0 1 0 1 1 1 1 1 1 0.0034500000000000004 > Run_info/SYS_400_RUN_69.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 70  0 1 0 1 1 1 1 1 1 0.0035 > Run_info/SYS_400_RUN_70.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 71  0 1 0 1 1 1 1 1 1 0.00355 > Run_info/SYS_400_RUN_71.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 72  0 1 0 1 1 1 1 1 1 0.0036000000000000003 > Run_info/SYS_400_RUN_72.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 73  0 1 0 1 1 1 1 1 1 0.00365 > Run_info/SYS_400_RUN_73.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 74  0 1 0 1 1 1 1 1 1 0.0037 > Run_info/SYS_400_RUN_74.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 75  0 1 0 1 1 1 1 1 1 0.0037500000000000003 > Run_info/SYS_400_RUN_75.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 76  0 1 0 1 1 1 1 1 1 0.0038 > Run_info/SYS_400_RUN_76.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 77  0 1 0 1 1 1 1 1 1 0.00385 > Run_info/SYS_400_RUN_77.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 78  0 1 0 1 1 1 1 1 1 0.0039000000000000003 > Run_info/SYS_400_RUN_78.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 79  0 1 0 1 1 1 1 1 1 0.00395 > Run_info/SYS_400_RUN_79.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 80  0 1 0 1 1 1 1 1 1 0.004 > Run_info/SYS_400_RUN_80.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 81  0 1 0 1 1 1 1 1 1 0.00405 > Run_info/SYS_400_RUN_81.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 82  0 1 0 1 1 1 1 1 1 0.0041 > Run_info/SYS_400_RUN_82.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 83  0 1 0 1 1 1 1 1 1 0.00415 > Run_info/SYS_400_RUN_83.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 84  0 1 0 1 1 1 1 1 1 0.004200000000000001 > Run_info/SYS_400_RUN_84.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 85  0 1 0 1 1 1 1 1 1 0.00425 > Run_info/SYS_400_RUN_85.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 86  0 1 0 1 1 1 1 1 1 0.0043 > Run_info/SYS_400_RUN_86.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 87  0 1 0 1 1 1 1 1 1 0.004350000000000001 > Run_info/SYS_400_RUN_87.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 88  0 1 0 1 1 1 1 1 1 0.0044 > Run_info/SYS_400_RUN_88.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 89  0 1 0 1 1 1 1 1 1 0.00445 > Run_info/SYS_400_RUN_89.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 90  0 1 0 1 1 1 1 1 1 0.0045000000000000005 > Run_info/SYS_400_RUN_90.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 91  0 1 0 1 1 1 1 1 1 0.00455 > Run_info/SYS_400_RUN_91.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 92  0 1 0 1 1 1 1 1 1 0.0046 > Run_info/SYS_400_RUN_92.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 93  0 1 0 1 1 1 1 1 1 0.0046500000000000005 > Run_info/SYS_400_RUN_93.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 94  0 1 0 1 1 1 1 1 1 0.0047 > Run_info/SYS_400_RUN_94.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 95  0 1 0 1 1 1 1 1 1 0.00475 > Run_info/SYS_400_RUN_95.txt &
wait 
python3 ocdeepDMD_Sequential.py '/gpu:0' 400 96  0 1 0 1 1 1 1 1 1 0.0048000000000000004 > Run_info/SYS_400_RUN_96.txt &
python3 ocdeepDMD_Sequential.py '/gpu:1' 400 97  0 1 0 1 1 1 1 1 1 0.00485 > Run_info/SYS_400_RUN_97.txt &
python3 ocdeepDMD_Sequential.py '/gpu:2' 400 98  0 1 0 1 1 1 1 1 1 0.0049 > Run_info/SYS_400_RUN_98.txt &
python3 ocdeepDMD_Sequential.py '/gpu:3' 400 99  0 1 0 1 1 1 1 1 1 0.00495 > Run_info/SYS_400_RUN_99.txt &
wait 
wait 
echo "All sessions are complete" 
echo "=======================================================" 
cd .. 
rm -R _current_run_saved_files 
rm -R Run_info 
cp -a oc_deepDMD/_current_run_saved_files/. _current_run_saved_files 
cp -a oc_deepDMD/Run_info/ Run_info 
cd oc_deepDMD/ 
