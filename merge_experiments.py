from argparse import ArgumentParser
import json, logging, os, time
import subprocess
import pandas as pd
import concurrent.futures
import urllib.request



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser()
    parser.add_argument("--results", help="results dir")
    parser.add_argument("--output", help="merged output csv path")
    args = parser.parse_args()

    df_list = []
    # merging csv files
    for file_name in os.listdir(args.results):
        if file_name.endswith(".csv"):
            df = pd.read_csv(os.path.join(args.results, file_name))
            if len(df[df['failed']==False]) == 0:
                df_list.append(df.tail(1))
            else:
                df_list.append(df[df['failed']==False])

    pd.concat(df_list).to_csv(args.output, index=False)