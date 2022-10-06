import os
from tqdm import tqdm
import json
from pathlib import Path
import librosa

from Parsers.interface import BaseRawParser, BasePreprocessor
from text import clean_text
from text.cleaners import check_twn
from .parser import DataParser
from preprocessing import preprocess_func
from preprocessing.preprocess_func_mp import *


class TATTTSRawParser(BaseRawParser):
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
        spker_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        print(f'[INFO] {len(spker_dirs)} speakers were found in wav dir : "{self.root}"')
        for spker_dir in tqdm(spker_dirs, position=0):
            res["all_speakers"].append(spker_dir.name)
            for wav_file in tqdm(spker_dir.rglob("*.wav"), position=1, leave=False):
                wav_path = str(wav_file)
                info_file = str(wav_file)[:-4] + '.json'
                with open(info_file, "r", encoding="utf-8") as f:
                    info = json.load(f)
                text = info["台羅數字調"]
                text = clean_text(text, ["tat_cleaners"])
                if not check_twn(text):
                    continue

                data = {
                    "wav_path": wav_path,
                    "text": text,
                }
                data_info = {
                    "spk": spker_dir.name,
                    "basename": f"{spker_dir.name}-{wav_file.name[:-4]}",
                }
                res["data"].append(data)
                res["data_info"].append(data_info)

        with open(self.data_parser.metadata_path, "w", encoding="utf-8") as f:
            json.dump(res["data_info"], f, indent=4)
        with open(self.data_parser.speakers_path, "w", encoding="utf-8") as f:
            json.dump(res["all_speakers"], f, indent=4)

        n = len(res["data_info"])
        tasks = list(zip(res["data_info"], res["data"], [False] * n))
        
        with Pool(processes=n_workers) as pool:
            for res in tqdm(pool.imap(ImapWrapper(self.prepare_initial_features), tasks, chunksize=64), total=n):
                pass


class TATTTSPreprocessor(BasePreprocessor):
    def __init__(self, preprocessed_root: Path):
        super().__init__(preprocessed_root)
        self.root = preprocessed_root
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

            text = self.data_parser.text.read_from_query(query)
            if not check_twn(text):
                continue
            
            if link_file.exists():
                os.unlink(str(link_file))
            if txt_link_file.exists():
                os.unlink(str(txt_link_file))
            os.link(wav_file, str(link_file))
            os.link(txt_file, str(txt_link_file))

    def mfa(self, mfa_data_dir: Path):
        corpus_directory = str(mfa_data_dir)
        dictionary_path = "lexicon/taiwanese.txt"
        acoustic_model_path = "MFA/TAT/taiwanese_acoustic_model.zip"
        output_directory = str(self.root / "TextGrid")
        cmd = f"mfa align {corpus_directory} {dictionary_path} {acoustic_model_path} {output_directory} -j 8 -v --clean"
        os.system(cmd)
    
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
