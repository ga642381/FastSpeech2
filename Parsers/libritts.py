import os
from tqdm import tqdm
from pathlib import Path
import librosa

from Parsers.interface import BaseRawParser, BasePreprocessor
from .parser import DataParser
from preprocessing import preprocess_func


class LibriTTSRawParser(BaseRawParser):
    def __init__(self, root: Path, preprocessed_root: Path):
        super().__init__(root)
        self.data_parser = DataParser(str(preprocessed_root))
        self.dsets = [
            "train-clean-100",
            "train-clean-360",
            "train-other-500",
            "dev-clean",
            "dev-other",
            "test-clean",
            "test-other",
        ]

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


class LibriTTSPreprocessor(BasePreprocessor):
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
        corpus_directory = str(mfa_data_dir)
        dictionary_path = "lexicon/librispeech-lexicon.txt"
        output_directory = str(self.root / "TextGrid")
        cmd = f"mfa align {corpus_directory} {dictionary_path} english {output_directory} -j 8 -v --clean"
        os.system(cmd)
    
    def denoise(self):
        pass

    def create_dataset(self):
        pass
