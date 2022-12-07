#!/bin/bash

file_name=voting
contract_name=Voting
function_name="winningProposal"

cd src/
python synthesizer.py \
    --root_dir ../illustration_examples/${file_name} \
    --file_name ${file_name}.sol \
    --contract_name ${contract_name} \
    --function_name ${function_name} \
    --output_file ${file_name}_${contract_name}_${function_name}.out \
    --use_const_cnt 1 --print_verbose

cd -