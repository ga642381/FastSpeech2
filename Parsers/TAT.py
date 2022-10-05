from codecs import ignore_errors
import os
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import librosa

from dlhlp_lib.audio import AUDIO_CONFIG

from Parsers.interface import BaseRawParser, BasePreprocessor
from text import clean_text
from text.cleaners import check_twn
from .parser import DataParser
from preprocessing import preprocess_func
from preprocessing.preprocess_func_mp import *


class TATRawParser(BaseRawParser):
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
        for dset in ["TAT-Vol1-train", "TAT-Vol2-train", "TAT-Vol1-eval", "TAT-Vol1-test", "TAT-Vol2-eval", "TAT-Vol2-test"]:
            wav_dir = self.root /  dset / "condenser" / "wav"
            if dset in ["TAT-Vol1-train", "TAT-Vol2-train"]:
                info_dir = self.root /  dset / "json"
            else:
                info_dir = self.root /  f"{dset}-key" / "json"
            spker_dirs = [d for d in wav_dir.iterdir() if d.is_dir()]
            print(f'[INFO] {len(spker_dirs)} speakers were found in wav dir : "{wav_dir}"')
            for spker_dir in tqdm(spker_dirs, position=0):
                res["all_speakers"].append(spker_dir.name)
                for wav_file in tqdm(spker_dir.rglob("*.wav"), position=1, leave=False):
                    wav_path = str(wav_file)
                    info_file = f"{str(info_dir)}/{spker_dir.name}/{wav_file.name[:-7]}.json"
                    with open(info_file, "r", encoding="utf-8") as f:
                        info = json.load(f)
                    text = info["台羅數字調"]
                    text = clean_text(text, ["tat_cleaners"])
                    if not check_twn(text):  # ignore data with chinese characters or unknown characters...
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
        return res


class TATPreprocessor(BasePreprocessor):
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
            wav_file = self.data_parser.wav_16000_enhanced.read_filename(query, raw=True)
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
        preprocess_func.denoise(self.root / "wav_16000", self.root / "wav_16000_enhanced")

    def create_dataset(self):
        INV_FRAME_PERIOD = AUDIO_CONFIG["audio"]["sampling_rate"] / AUDIO_CONFIG["stft"]["hop_length"]
        queries = self.data_parser.get_all_queries()
        textgrid2segment_and_phoneme_mp(
            self.data_parser, queries,
            "textgrid", "mfa_segment", "phoneme",
            n_workers=os.cpu_count() // 2,
            ignore_errors=False
        )
        # trim_wav_by_segment_mp(
        #     self.data_parser, queries, 22050, 
        #     "wav_22050_enhanced", "mfa_segment", "wav_trim_22050_enhanced",
        #     refresh=True,
        #     n_workers=2,
        #     ignore_errors=True
        # )
        # wav_to_mel_energy_pitch_mp(
        #     self.data_parser, queries,
        #     "wav_trim_22050_enhanced", "mel", "energy", "pitch", "interpolate_pitch",
        #     n_workers=4,
        #     ignore_errors=True
        # )
        # segment2duration_mp(
        #     self.data_parser, queries, INV_FRAME_PERIOD,
        #     "mfa_segment", "mfa_duration",
        #     refresh=True,
        #     n_workers=os.cpu_count() // 2,
        #     ignore_errors=True
        # )
        # duration_avg_pitch_and_energy_mp(
        #     self.data_parser, queries,
        #     "mfa_duration", "interpolate_pitch", "energy",
        #     refresh=True,
        #     n_workers=os.cpu_count() // 2,
        #     ignore_errors=True
        # )
