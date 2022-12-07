#!/bin/bash

file_name=nestedloop
contract_name=NestedLoop
function_name="main"

cd src

python synthesizer.py \
    --root_dir ../illustration_examples/${file_name} \
    --file_name ${file_name}.sol \
    --contract_name ${contract_name} \
    --function_name ${function_name} \
    --output_file ${file_name}_${contract_name}_${function_name}.out \
    --use_quadratic --is_nestedloop --use_const_cnt 1 --print_verbose

rm *.smt2 *.model *.gas
cd -