import random
import json
from tqdm import tqdm

from Parsers.parser import DataParser
from config import hparams as hp


def split_dataset(data_parser: DataParser):
    random.seed(829)
    queries = data_parser.get_all_queries()
    random.shuffle(queries)

    # train_split, val_split, test_split = [], [], []
    res = []
    for query in tqdm(queries):
        try:
            mel = data_parser.mel.read_from_query(query)
            assert mel.shape[1] <= hp.max_seq_len and mel.shape[1] >= hp.min_seq_len
            res.append(query)
        except:
            continue  # file does not exist or audio too long / too short
    
    n = len(res)
    train_split = res[:-int(0.1 * n)]
    valid_split = res[-int(0.1 * n):]
        
    ### Save data ###
    with open(f"{data_parser.root}/metadata.json", "w", encoding="utf-8") as f:
        data = {"train": train_split, "valid": valid_split}
        json.dump(data, f, indent=4)
    