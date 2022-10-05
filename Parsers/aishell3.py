import os
from tqdm import tqdm
from pathlib import Path
import librosa

from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from preprocessing import preprocess_func


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

    def parse(self):
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
        return res
