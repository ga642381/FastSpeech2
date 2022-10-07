import os
from tqdm import tqdm
import json
from pathlib import Path
import librosa

from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from preprocessing import preprocess_func
from preprocessing.preprocess_func_mp import *


class AISHELL3RawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        wav_16000, _ = librosa.load(data["wav_path"], sr=16000)
        wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.wav_22050.save(wav_22050, query)
        self.data_parser.text.save(data["text"], query)

    def parse(self, n_workers=4):
        res = {"data": [], "data_info": [], "all_speakers": []}

        # train
        path = f"{self.root}/train/label_train-set.txt"
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in tqdm(enumerate(f)):
                if i < 5 or line == '\n':
                    continue
                wav_name, text, _ = line.strip().split('|')
                speaker = wav_name[:-4]
                if speaker not in res["all_speakers"]:
                    res["all_speakers"].append(speaker)
                wav_path = f"{self.root}/train/wav/{speaker}/{wav_name}.wav"
                if os.path.isfile(wav_path):
                    data = {
                        "wav_path": wav_path,
                        "text": text,
                    }
                    data_info = {
                        "spk": speaker,
                        "basename": wav_name,
                        "dset": "train",
                    }
                    res["data"].append(data)
                    res["data_info"].append(data_info)
                else:
                    print("transcript.txt should not contain non-exist wav files, data might be corrupted.")
                    print(f"Can not find {wav_path}.")
        
        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(res["data_info"], f, indent=4)
        with open(self.data_parser.speakers_path, "w", encoding="utf-8") as f:
            json.dump(res["all_speakers"], f, indent=4)

        n = len(res["data_info"])
        tasks = list(zip(res["data_info"], res["data"], [False] * n))
        
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(self.prepare_initial_features), tasks, chunksize=64), total=n):
                pass


class AISHELL3Preprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_mfa(self, mfa_data_dir: Path):
        queries = self.data_parser.get_all_queries()

        # 1. create a similar structure in "mfa_data_dir" as in "wav_dir"
        for spk in self.data_parser.get_all_speakers():
            target_dir = mfa_data_dir / spk
            target_dir.mkdir(parents=True, exist_ok=True)

        # 2. create hard link for wav file
        for query in tqdm(queries):
            target_dir = mfa_data_dir / query['spk']
            link_file = target_dir / f"{query['spk']}-{query['basename']}.wav"
            txt_link_file = target_dir / f"{query['spk']}-{query['basename']}.txt"
            wav_file = self.data_parser.wav_16000.read_filename(query, raw=True)
            txt_file = self.data_parser.text.read_filename(query, raw=True)
            
            if link_file.exists():
                os.unlink(str(link_file))
            if txt_link_file.exists():
                os.unlink(str(txt_link_file))
            os.link(wav_file, str(link_file))
            os.link(txt_file, str(txt_link_file))

    def mfa(self, mfa_data_dir: Path):
        # TODO: provide mfa checkpoint
        corpus_directory = str(mfa_data_dir)
        dictionary_path = "lexicon/pinyin-lexicon-r.txt"
        output_directory = str(self.root / "TextGrid")
        # cmd = f"mfa align {corpus_directory} {dictionary_path} english {output_directory} -j 8 -v --clean"
        # os.system(cmd)
    
    def denoise(self):
        pass

    def create_dataset(self):
        queries = self.data_parser.get_all_queries()
        process_utterance_mp(
            self.data_parser, queries,
            "wav_22050",
            n_workers=8, chunksize=64,
            ignore_errors=True
        )
