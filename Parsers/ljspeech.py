import os
from tqdm import tqdm
from pathlib import Path
import librosa

from dlhlp_lib.parsers.Interfaces import BaseRawParser

from .parser import DataParser


class LJSpeechRawParser(object):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.root = root
        self.data_parser = DataParser(str(preprocessed_root))

    def prepare_initial_features(self, query, data):
        wav_16000, _ = librosa.load(data["wav_path"], sr=16000)
        wav_22050, _ = librosa.load(data["wav_path"], sr=22050)
        self.data_parser.wav_16000.save(wav_16000, query)
        self.data_parser.wav_22050.save(wav_22050, query)
        self.data_parser.text.save(data["text"], query)

    def parse(self):
        res = {"data": [], "data_info": [], "all_speakers": []}
        for dset in self.dsets:
            if not os.path.isdir(f"{self.root}/{dset}"):
                continue
            for speaker in tqdm(os.listdir(f"{self.root}/{dset}"), desc=dset):
                res["all_speakers"].append(speaker)
                for chapter in os.listdir(f"{self.root}/{dset}/{speaker}"):
                    for filename in os.listdir(f"{self.root}/{dset}/{speaker}/{chapter}"):
                        if filename[-4:] != ".wav":
                            continue
                        basename = filename[:-4]
                        wav_path = f"{self.root}/{dset}/{speaker}/{chapter}/{basename}.wav"
                        text_path = f"{self.root}/{dset}/{speaker}/{chapter}/{basename}.normalized.txt"
                        with open(text_path, "r", encoding="utf-8") as f:
                            text = f.readline().strip("\n")
                        data = {
                            "wav_path": wav_path,
                            "text": text,
                        }
                        data_info = {
                            "spk": speaker,
                            "basename": basename,
                            "dset": dset,
                            "chapter": chapter,
                        }
                        res["data"].append(data)
                        res["data_info"].append(data_info)
        return res


import os
from tqdm import tqdm
import json
from pathlib import Path
import librosa

from Parsers.interface import BaseRawParser
from text import clean_text
from text.cleaners import check_twn
from .parser import DataParser
from preprocessing import preprocess_func


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
        
    def parse(self):
        res = {"data": [], "data_info": [], "all_speakers": []}
        spker_dirs = [d for d in self.root.iterdir() if d.is_dir()]
        print(f'[INFO] {len(spker_dirs)} speakers were found in wav dir : "{self.root}"')
        for spker_dir in tqdm(spker_dirs):
            res["all_speakers"].append(spker_dir.name)
            for wav_file in tqdm(spker_dir.rglob("*.wav")):
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
                    "basename": wav_file.name[:-4],
                }
                res["data"].append(data)
                res["data_info"].append(data_info)

        return res


class TATTTSPreprocessor(object):
    def __init__(self, preprocessed_root: Path):
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
        cmd = f"mfa align {corpus_directory} {dictionary_path} {acoustic_model_path} {output_directory} -j 8 -v"
        os.system(cmd)
    
    def denoise(self):
        preprocess_func.denoise(self.root / "wav_16000", self.root / "wav_16000_enhanced")

    def create_dataset(self):
        pass
