import os
from multiprocessing import Pool, set_start_method
from tqdm import tqdm
import json
from pathlib import Path

from Parsers.interface import BaseRawParser
from Parsers.TAT_TTS import TATTTSRawParser
from Parsers.TAT import TATRawParser


def preprocess_func(raw_parser: BaseRawParser, query, data):
    raw_parser.prepare_initial_features(query, data)


def imap_preprocess_func(task):
    preprocess_func(*task)


def preprocess_raw(parser_name, raw_root, preprocessed_root, n_workers=4):
    os.makedirs(preprocessed_root, exist_ok=True)
    print(f"Parsing raw data from {raw_root}...")
    if parser_name == "TATTTS":
        raw_parser = TATTTSRawParser(Path(raw_root), Path(preprocessed_root))
    elif parser_name == "TAT":
        raw_parser = TATRawParser(Path(raw_root), Path(preprocessed_root))
    else:
        raise NotImplementedError

    res = raw_parser.parse()
    data_infos = res["data_info"]
    datas = res["data"]

    with open(f"{preprocessed_root}/data_info.json", "w", encoding="utf-8") as f:
        json.dump(res["data_info"], f, indent=4)

    with open(f"{preprocessed_root}/speakers.json", "w", encoding="utf-8") as f:
        json.dump(res["all_speakers"], f, indent=4)

    n = len(data_infos)
    tasks = list(zip([raw_parser] * n, data_infos, datas))
    
    with Pool(processes=n_workers) as pool:
        for i in tqdm(pool.imap(imap_preprocess_func, tasks, chunksize=64), total=n):
            pass


if __name__ == "__main__":
    from sys import platform
    if platform == "linux" or platform == "linux2":
        set_start_method("spawn", force=True)
    # preprocess_raw("AISHELL-3", "/work/Data/AISHELL-3", "./preprocessed_data/AISHELL-3")
    # preprocess_raw("CSS10", "/work/Data/CSS10/german", "./preprocessed_data/CSS10/german")
    # preprocess_raw("JSUT", "/work/Data/jsut_ver1.1", "./preprocessed_data/JSUT")
    # preprocess_raw("KSS", "/work/Data/kss", "./preprocessed_data/kss")
    # preprocess_raw("LibriTTS", "/work/Data/LibriTTS", "./preprocessed_data/LibriTTS")
    # preprocess_raw("GlobalPhone", "/work/Data/GlobalPhone/French", "./preprocessed_data/GlobalPhone/french")
    preprocess_raw("TATTTS", "/mnt/d/Data/TAT-TTS", "./preprocessed/TAT-TTS")
    preprocess_raw("TAT", "/mnt/d/Data/TAT", "./preprocessed/TAT")
