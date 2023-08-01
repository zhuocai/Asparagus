#!/bin/bash
#
# Call run_experiments.py that outputs gas bounds for the complete gastap dataset. 
# 
# For each function, run_experiments.py tries three options. 

python3 run_experiments.py --dataset dataset/gastap_dataset --output experiments/results/ --max-workers 16 --timeout 25 --function-list dataset/gastap_all_functions.csv

rm *.smt2 *.model *.gas

python3 merge_experiments.py --results experiments/results/ --output experiment_results/asparagus_run_all.csv