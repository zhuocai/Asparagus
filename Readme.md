# Asparagus: Automated Synthesis of Parametric Gas Upper-bounds for Smart Contracts

### Illustration Examples
Two examples, voting (linear) and nestedloop (quadratic) are provided in illustration_examples folder.  
Run the following commands to synthesize bounds. 
```bash
bash illustration_examples/run_voting.sh
```
```bash
bash illustration_examples/run_nestedloop.sh
```

For example, if we execute run_voting.sh, the last two lines of terminal output should be similar to the following:
```
366.0000 + 1324.0000*g__proposals__
Time Used: 1.2097830772399902
```
Note: g__proposals__ indicates the length of array proposals.  
### GASTAP Dataset
The GASTAP dataset is in dataset/gastap_dataset folder. Each subfolder contains a solidity source code file (.sol file).  
Each contract in the solidity file is compiled and converted into a RBR file using the EthIR tool.  
We also generate a file (.meta) for each solidity file that stores auxiliary information about rbr and variables.  

* dataset/gastap_all_functions.csv: basic information of the 156735 functions.  
* experiment_results/10186_case_info_analyzed.csv: analyzed experimental results of 10186 functions, where GASTAP estimation results are available for comparison.  
* experiment_results/merged_case_info_analyzed.csv: analyzed experimental results on full gastap dataset.  

### Run a Single Example
```bash
bash run_example.sh
```
Replace the file_name, contract_name, function_name to run other examples.  
Choose one of three recommended options. Check run_example.sh for more information.  


### Run Complete Experiments
We also provide run_experiments.py and run_all.sh to automatically synthesize gas bounds for the complete GASTAP dataset.  
```bash
bash run_all.sh
```

To run experiments for a subset, you can select some rows from dataset/gastap_all_functions.csv.  

