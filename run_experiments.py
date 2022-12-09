from argparse import ArgumentParser
import json, logging, os, time
import subprocess
import pandas as pd
import concurrent.futures
import urllib.request

def do_experiment_for_record(
        file_name, 
        contract_name, 
        function_name,
        record_dir,
        output_dir,
        timeout,
        extra_args
        
    ):

    contract_address =  file_name.replace(".sol", "")


    result = {
        'start_time': int(time.time()),
        'end_time': None,
        'function': function_name,
        "contract": contract_name,
        "file": file_name,
        "gas_estimation": None,
        "error": "",
        "failed": True,
        "timeout": False
    }
    log_name = "{}_{}_{}_{}.err".format(
        file_name,
        contract_name,
        function_name,
        result['start_time']
    )
    output_file = os.path.join(output_dir, contract_address+".csv")

    # Linear + Z3
    try:
        logging.info(
            "{}: {}.{} -> Solving with {}".format(
                file_name,
                contract_name,
                function_name,
                extra_args
            )
        )
        subprocess.run(
            [
                'python', "-u", "gas-new/synthesizer.py",
                "--root_dir", record_dir,
                "--file_name", file_name,
                "--contract_name", contract_name,
                "--function_name", function_name,
                "--output_file", output_file
            ]+extra_args,
            stdout=subprocess.DEVNULL,
            stderr=open(os.path.join(output_dir, log_name), 'a'),
            timeout=int(timeout),
            check=True
        )
        return True
    except subprocess.TimeoutExpired as e:
        result['timeout'] = True
        logging.error(
            "{}: {}.{} -> Timeout with {}".format(
                file_name,
                contract_name,
                function_name,
                extra_args
            )
        )
    except Exception as e:
        result['timeout'] = False
        result['error'] = str(e)
        logging.error(
            "{}: {}.{} -> Faild with {}:{}".format(
                file_name,
                contract_name,
                function_name,
                extra_args,
                str(e)
            )
        )
    result['end_time'] = int(time.time())
    pd.DataFrame([result]).to_csv(
        output_file,
        mode='a',
        header=not os.path.exists(output_file),
        index=False
    )
    return False


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--dataset", help="dataset dir")
    parser.add_argument("--output", help="output dir")
    parser.add_argument("--function-list", help="list of function")
    parser.add_argument("--timeout", help="timeout", default=60*15)
    parser.add_argument("--max-workers", help="the number of maximum workers", default=1)
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    if args.function_list:
        df = pd.read_csv(args.function_list)
    else:
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

    solved = {}

    for extra_args in [
            [
                "--use_const_cnt", "1", 
                "--use_const_gas"
                ],
            [
                "--use_const_cnt", "1"
                ],
            [
                "--use_const_cnt", "2"
                ],
        ]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            
            future_to_key = {}
            
            for i, row in df.iterrows():
            
                key = "{file_name}_{contract_name}_{function_name}".format(
                    file_name=row['file'],
                    contract_name=row['contract'],
                    function_name=row['function'],
                )
            
                if key in solved:
                    if solved[key]:
                        continue
                
                # Original function call
                # solved[key] = do_experiment_for_record(
                #     file_name=row['file'],
                #     contract_name=row['contract'],
                #     function_name=row['function'],
                #     record_dir=os.path.join(args.dataset, row['contract_address']),
                #     output_dir=args.output,
                #     timeout=args.timeout,
                #     extra_args=extra_args
                # )
                
                future_to_key[
                    executor.submit(
                            do_experiment_for_record, 
                            file_name=row['file'],
                            contract_name=row['contract'],
                            function_name=row['function'],
                            record_dir=os.path.join(args.dataset, row['file'].replace(".sol", "")),
                            output_dir=args.output,
                            timeout=args.timeout,
                            extra_args=extra_args
                        )
                    ] = key
            
            for future in concurrent.futures.as_completed(future_to_key):
                key = future_to_key[future]
                try:
                    solved[key] = future.result()
                except Exception as e:
                    logging.exception(e)
        