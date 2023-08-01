#!/bin/bash
# options:
# 1. --use_const_gas --use_const_cnt 1
# 2. --use_const_cnt 1
# 3. --use_const_cnt 
# Use option 1 first: the gas bound is restricted to const expression
# If option 1 does not work, use option 2. A parametric gas bound is allowed. use_const_cnt=1 means each cutset node has at most 1 symbolic (unknown) invariant. 
# If option 2 does not work, use option 3. Each cutset node has at most 2 symbolic unvariants. 

file_name=0x69583c6f3d60786e7fd4054c2d539771c36944c1 # CHANGE THIS
contract_name=token # CHANGE THIS
function_name="transfer(address,uint256)" # CHANGE THIS

cd src/
python3 synthesizer.py \
    --root_dir ../dataset/gastap_dataset/${file_name} \
    --file_name ${file_name}.sol \
    --contract_name ${contract_name} \
    --function_name ${function_name} \
    --output_file ../experiments/results/run_example.csv \
    --print_verbose \
    --use_const_cnt 1 # CHANGE THIS: replace this line with other options when necessary

rm *.smt2 *.model *.gas

cd -
