import os
import json
from typing import Dict, List

from dlhlp_lib.parsers.Interfaces import BaseDataParser
from dlhlp_lib.parsers.Feature import Feature
from dlhlp_lib.parsers.QueryParsers import SFQueryParser, NestSFQueryParser
from dlhlp_lib.parsers.IOObjects import NumpyIO, PickleIO, WavIO, TextGridIO, TextIO


class DataParser(BaseDataParser):
    """
    Basic FastSpeech2 features + speech enhancement features.
    """

    def __init__(self, root: str):
        super().__init__(root)

        self.wav_16000 = Feature(
            "wav_16000", root, SFQueryParser(f"{self.root}/wav_16000"), WavIO(sr=16000))
        self.wav_16000_enhanced = Feature(
            "wav_16000_enhanced", root, SFQueryParser(f"{self.root}/wav_16000_enhanced"), WavIO(sr=16000))
        self.wav_22050 = Feature(
            "wav_22050", root, SFQueryParser(f"{self.root}/wav_22050"), WavIO(sr=22050))
        self.wav_22050_enhanced = Feature(
            "wav_22050_enhanced", root, SFQueryParser(f"{self.root}/wav_22050_enhanced"), WavIO(sr=22050))
        
        self.mfa_segment = Feature(
            "mfa_segment", root, NestSFQueryParser(f"{self.root}/mfa_segment"), PickleIO(), enable_cache=True)
        self.textgrid = Feature(
            "TextGrid", root, NestSFQueryParser(f"{self.root}/TextGrid"), TextGridIO())
        self.phoneme = Feature(
            "phoneme", root, NestSFQueryParser(f"{self.root}/phoneme"), TextIO(), enable_cache=True)
        self.text = Feature(
            "text", root, NestSFQueryParser(f"{self.root}/text"), TextIO(), enable_cache=True)

        self.wav_trim_22050 = Feature(
            "wav_trim_22050", root, NestSFQueryParser(f"{self.root}/wav_trim_22050"), NumpyIO())
        self.wav_trim_22050_enhanced = Feature(
            "wav_trim_22050_enhanced", root, NestSFQueryParser(f"{self.root}/wav_trim_22050_enhanced"), NumpyIO())
        self.mel = Feature(
            "mel", root, NestSFQueryParser(f"{self.root}/mel"), NumpyIO())
        self.pitch = Feature(
            "pitch", root, NestSFQueryParser(f"{self.root}/pitch"), NumpyIO(), enable_cache=True)
        self.interpolate_pitch = Feature(
            "interpolate_pitch", root, NestSFQueryParser(f"{self.root}/interpolate_pitch"), NumpyIO(), enable_cache=True)
        self.energy = Feature(
            "energy", root, NestSFQueryParser(f"{self.root}/energy"), NumpyIO(), enable_cache=True)
        self.mfa_duration = Feature(
            "mfa_duration", root, NestSFQueryParser(f"{self.root}/alignment"), NumpyIO(), enable_cache=True)
        self.mfa_duration_avg_pitch = Feature(
            "mfa_duration_avg_pitch", root, NestSFQueryParser(f"{self.root}/alignment_avg_pitch"), NumpyIO(), enable_cache=True)
        self.mfa_duration_avg_energy = Feature(
            "mfa_duration_avg_energy", root, NestSFQueryParser(f"{self.root}/alignment_avg_energy"), NumpyIO(), enable_cache=True)
        
        self.stats_path = f"{self.root}/stats.json"
        self.speakers_path = f"{self.root}/speakers.json"

    def _init_structure(self):
        os.makedirs(f"{self.root}/wav_16000", exist_ok=True)
        os.makedirs(f"{self.root}/wav_22050", exist_ok=True)
        os.makedirs(f"{self.root}/wav_16000_enhanced", exist_ok=True)
        os.makedirs(f"{self.root}/wav_22050_enhanced", exist_ok=True)

        os.makedirs(f"{self.root}/mfa_segment", exist_ok=True)
        os.makedirs(f"{self.root}/phoneme", exist_ok=True)
        os.makedirs(f"{self.root}/text", exist_ok=True)

        os.makedirs(f"{self.root}/wav_trim_22050", exist_ok=True)
        os.makedirs(f"{self.root}/wav_trim_22050_enhanced", exist_ok=True)
        os.makedirs(f"{self.root}/mel", exist_ok=True)
        os.makedirs(f"{self.root}/pitch", exist_ok=True)
        os.makedirs(f"{self.root}/interpolate_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/energy", exist_ok=True)
        os.makedirs(f"{self.root}/alignment", exist_ok=True)
        os.makedirs(f"{self.root}/alignment_avg_pitch", exist_ok=True)
        os.makedirs(f"{self.root}/alignment_avg_energy", exist_ok=True)
    
    def get_all_queries(self):
        with open(f"{self.root}/data_info.json", "r", encoding="utf-8") as f:
            data_infos = json.load(f)
        return data_infos

    def get_feature(self, query: str) -> Feature:
        return getattr(self, query)

    def get_all_speakers(self) -> List[str]:
        with open(self.speakers_path, 'r', encoding='utf-8') as f:
            speakers = json.load(f)
        return speakers
