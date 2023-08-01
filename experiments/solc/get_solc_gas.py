from argparse import ArgumentParser
import json, logging, os, time
import subprocess
import pandas as pd
from tqdm import tqdm

COSTABS_PATH = '/tmp/costabs'

def get_solc_gas(file_name):
    function_list = []
    solc_output = subprocess.check_output(
        [
            "solc",
            "--gas",
            file_name,
        ],
        stderr=subprocess.DEVNULL
    )
    solc_output = solc_output.decode("utf-8") 

    current_contract = None
    for line in solc_output.split("\n"):
        if "=======" in line:
            current_contract = line.replace("=======", "").split(":")[-1].strip()
        elif "(" in line and ":" in line:
            function_list.append(
                {
                    'file': file_name,
                    'contract': current_contract,
                    'function': line.split(":")[0].strip(),
                    "gas_estimation": line.split(":")[-1].strip()
                }
            )
    return function_list
    
        

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--dataset", help="dataset dir")
    parser.add_argument("--output", help="output file")
    args = parser.parse_args()


    dataset = []
    for contract_address in os.listdir(args.dataset):
        try:
            with open(os.path.join(args.dataset, contract_address, contract_address+".meta"), 'r') as f:
                meta = json.load(f)
            for contract in meta:
                for function_name in meta[contract]['function_block_mapping']:
                    dataset.append(
                        {
                            "contract_address": contract_address,
                            'file':contract_address+".sol",
                            'contract': contract,
                            "function": function_name
                        }
                    )
        except:
            pass
    df = pd.DataFrame(dataset)
    
    logging.info('running for {} funtions'.format(len(df)))

    solc_result = []
    for i, row in tqdm(df.drop_duplicates('file').iterrows()):
        try:
            solc_result.extend(
                get_solc_gas(
                    file_name=os.path.join(args.dataset, row['contract_address'], row['file'])
                    )
                )
        except Exception as e:
            logging.error(row)
            logging.exception(e)

        pd.DataFrame(solc_result).to_csv(
            args.output,
            index=False
        )