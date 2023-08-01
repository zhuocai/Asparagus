# -*- coding: utf-8 -*-
"""
  This is a script for fetching upper bounds from http://costa.fdi.ucm.es/gastap/ 
  for smart contracts.
"""

from random import random
from urllib.parse import unquote, quote_plus, quote 
import requests
import time, os, csv, logging, pandas, random
from bs4 import BeautifulSoup
from tqdm import tqdm
from argparse import ArgumentParser
import concurrent.futures


logging.basicConfig(level=logging.INFO)


# This function fetchs available methods in a smart contract. It work by sending a post request such as original website.
def fetch_methods(file_path):

  with open(file_path, "r") as f:
    original_code = str(f.read())+"\n"

  payload_temp = "data=%7B%22!_cmd_!%22%3A%22ethiroutline%22%2C%22!_files_num_!%22%3A1%2C%22!_files_scheme_!%22%3A%22first%22%2C%22!_files_pname_!%22%3Anull%2C%22!_file_name_!0%22%3A%22{name}%22%2C%22!_file_content_!0%22%3A%22{content}%22%2C%22!_result_media_!%22%3A%22stdin%22%7D"
  payload = payload_temp.format(
      name = os.path.basename(file_path),
      content=quote_plus(original_code, safe='\\()*\':').replace('%0A', '%5Cn')
      )
  url = "http://costa.fdi.ucm.es/gastap/ei/php/cmdengine.php"

  headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8,ru;q=0.7',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': 'http://costa.fdi.ucm.es',
    'Referer': 'http://costa.fdi.ucm.es/gastap/',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
  }

  response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
  soup = BeautifulSoup(response.text, 'html.parser')
  methods = []
  for c in soup.findAll('class'):
    for m in c.findAll('method'):
      methods.append('{}.{}'.format(c['name'], m['name']))

  return methods

# Fetch memory and opcode upper bound for each method 
def fetch_upper_bounds(file_path, method):
  with open(file_path, "r") as f:
      original_code = str(f.read())+"\n"

  payload_temp = "data=%7B%22!_cmd_!%22%3A%22ethir%22%2C%22-ethir_mem%22%3A%22+yes%22%2C%22-type_file%22%3A%22+solidity%22%2C%22!_files_num_!%22%3A1%2C%22!_files_scheme_!%22%3A%22first%22%2C%22!_files_pname_!%22%3A%22-undefined%22%2C%22!_file_name_!0%22%3A%22{name}%22%2C%22!_file_content_!0%22%3A%22{content}%22%2C%22!_entries_scheme_!%22%3A%22param%22%2C%22!_entries_pname_!%22%3A%22-entries%22%2C%22!_entries_content_!%22%3A%22+{method}%22%2C%22!_result_media_!%22%3A%22file%22%2C%22!_result_filename_!%22%3A%22%2Ftmp%2Fcostabs%2Foutput.xml%22%7D"
  payload = payload_temp.format(
      name = os.path.basename(file_path),
      content=quote_plus(original_code, safe='\\()*\':').replace('%0A', '%5Cn'),
      method = method
      )

  import requests

  url = "http://costa.fdi.ucm.es/gastap/ei/php/cmdengine.php"

  headers = {
    'Accept': '*/*',
    'Accept-Language': 'en-US,en;q=0.9,fa;q=0.8,ru;q=0.7',
    'Connection': 'keep-alive',
    'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
    'Origin': 'http://costa.fdi.ucm.es',
    'Referer': 'http://costa.fdi.ucm.es/gastap/',
    'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.5005.61 Safari/537.36',
    'X-Requested-With': 'XMLHttpRequest'
  }

  response = requests.request("POST", url, headers=headers, data=payload, timeout=30)

  soup = BeautifulSoup(response.text, 'html.parser')
  result = {}

  for com in soup.findAll('eicommand', {'type': 'console'}):
    t = com.find('text').text

    if 'terminates' in t:
      result['terminates'] = True if t.split(":")[-1].strip() == 'yes' else False 
    elif 'Memory' in t:
      result['memory'] = t.split(":")[-1].strip()
    elif 'Opcodes' in t:
      result['opcode'] = t.split(":")[-1].strip()

  return result


if __name__ == "__main__":

  parser = ArgumentParser()
  parser.add_argument("--dataset", help="dataset dirs")
  parser.add_argument("--output", help="output dir")
  parser.add_argument("--max-workers", help="the number of maximum workers", default=1)
  args = parser.parse_args()

  smart_contract_dirs = args.dataset
  result_file = args.output

  processed_files = set(pandas.read_csv(result_file).file_name)
  logging.info("{} is processed".format(len(processed_files)))
  
  undone_files = [file_name for file_name in os.listdir(smart_contract_dirs) if file_name not in processed_files]

  with open(result_file, 'a') as csvfile:
    csvwriter = csv.writer(csvfile)

    for file_name in tqdm(undone_files):
      try:

        # do not process a file if we already did it
        if file_name in processed_files:
          continue

        file_path = os.path.join(smart_contract_dirs, file_name)

        # feching methods of the smart contract
        methods = fetch_methods(file_path)

        if not len(methods):
          csvwriter.writerow([file_name, None, None, None, None])
          logging.error(file_name + " is failed!")
        else:
          with concurrent.futures.ThreadPoolExecutor(max_workers=int(args.max_workers)) as executor:
            
            future_to_method = {}
            
            for m in methods:
                # upper_bounds = fetch_upper_bounds(file_path, m)
                future_to_method[
                    executor.submit(
                            fetch_upper_bounds, 
                            file_path,
                            m
                        )
                    ] = m
            
            for future in concurrent.futures.as_completed(future_to_method):
                method = future_to_method[future]
                try:
                    upper_bounds = future.result()
                    csvwriter.writerow([
                      file_name, 
                      method, 
                      upper_bounds['terminates'] if 'terminates' in upper_bounds else None, 
                      upper_bounds['memory'] if 'memory' in upper_bounds else None, 
                      upper_bounds['opcode'] if 'opcode' in upper_bounds else None, 
                    ])
                except Exception as e:
                    logging.error("Failed: {}: {}".format(file_name, method))
                    logging.exception(e)

      except Exception as e:
        logging.exception(e)
        csvwriter.writerow([file_name, None, None, None, None])

