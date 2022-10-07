from typing import List, Set
from tqdm import tqdm

from Parsers.parser import DataParser


def collect_phonemes(data_dirs) -> Set[str]:
    all_phns = set()
    for data_dir in data_dirs:
        data_parser = DataParser(data_dir)
        queries = data_parser.get_all_queries()
        for query in tqdm(queries):
            try:
                phn_seq = data_parser.phoneme.read_from_query(query)
                phn_seq = phn_seq.replace('{', '').replace('}', '')
                phn_seq = phn_seq.strip().split()
                all_phns.update(phn_seq)
            except:
                pass
    return all_phns


if __name__ == "__main__":
    phns = collect_phonemes([
        "preprocessed/TAT-TTS", 
        "preprocessed/TAT"
    ])
    phns = list(phns)
    phns.sort()
    with open("lexicon/taiwanese_phones.txt", 'w', encoding='utf-8') as f:
        for p in phns:
            f.write(f"{p}\n")
    print(phns)
    print(len(phns))
