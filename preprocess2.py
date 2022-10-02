import argparse
import os
import numpy as np
import torchaudio
from pathlib import Path
import shutil
import librosa
from tqdm import tqdm

from config import hparams as hp
from preprocessing.preprocess_raw import preprocess_raw
from Parsers import get_preprocessor


if hp.dataset in ["LibriTTS", "VCTK", "LJSpeech", "TAT", "TATTTS"]:
    from data import functional as F
else:
    raise NotImplementedError(
        "You should specify the dataset in hparams.py\
                               and write a corresponding file in data/"
    )
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class Preprocessor:
    def __init__(self, args):
        self.args = args
        self.dataset = hp.dataset
        self.root = args.raw_dir
        self.preprocessed_root = args.preprocessed_dir
        self.processor = get_preprocessor(hp.dataset)(Path(args.preprocessed_dir))

    def exec(self):
        self.print_message()
        key_input = ""
        while key_input not in ["y", "Y", "n", "N"]:
            key_input = input("Proceed? ([y/n])? ")

        if key_input in ["y", "Y"]:
            # 0. Initial features from raw data
            # self.prepare_initial_features()
            # 1. Denoising
            if self.args.denoise:
                print("[INFO] Preparing data for Montreal Force Alignmnet...")
                torchaudio.set_audio_backend("sox_io")
                self.processor.denoise()
            # 2. Prepare MFA
            if self.args.prepare_mfa:
                print("[INFO] Preparing data for Montreal Force Alignmnet...")
                self.processor.prepare_mfa(Path(self.root) / "mfa_data")
            # 3. MFA
            if self.args.mfa:
                print("[INFO] Performing Montreal Force Alignment...")
                self.processor.mfa(Path(self.root) / "mfa_data")
            # 4. Create Dataset
            if self.args.create_dataset:
                print("[INFO] Creating Training and Validation Dataset...")
                self.processor.create_dataset()

    def print_message(self):
        print("\n")
        print("------ Preprocessing ------")
        print(f"* Dataset     : {self.dataset}")
        print(f"* Raw Data path   : {self.root}")
        print(f"* Output path : {self.preprocessed_root}")
        print("\n")
        print(" [INFO] The following will be executed:")
        if self.args.denoise:
            print("* Denoising corpus")
        if self.args.prepare_mfa:
            print("* Preparing data for Montreal Force Alignment")
        if self.args.mfa:
            print("* Montreal Force Alignmnet")
        if self.args.create_dataset:
            print("* Creating Training Dataset")
        print("\n")

    def prepare_initial_features(self):
        preprocess_raw(hp.dataset, self.root, self.preprocessed_root)

    # === 4. Create Dataset === #
    def create_dataset(self):
        """
        * metadata.json will be created
        * mel, energy, f0,... will be created
        """
        in_dir = self.wav_dir
        out_dir = self.out_dir

        F.build_dataset(in_dir, out_dir)


def main(args):
    P = Preprocessor(args)
    P.exec()


if __name__ == "__main__":
    """
    e.g.
    # LJSpeech #
        * run ./script/organizeLJ.py first
        * python preprocess.py /storage/tts2021/LJSpeech-organized/wavs /storage/tts2021/LJSpeech-organized/txts ./processed/LJSpeech --prepare_mfa --mfa --create_dataset
    
    # LibriTTS #
        * python preprocess.py /storage/tts2021//LibriTTS/train-clean-360 /storage/tts2021//LibriTTS/train-clean-360 ./processed/LibriTTS --prepare_mfa --mfa --create_dataset
    
    # VCTK #
        * python preprocess.py /storage/tts2021/VCTK-Corpus/wav48/ /storage/tts2021/VCTK-Corpus/txt ./processed/VCTK --prepare_mfa --mfa --create_dataset
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_dir", type=str)
    parser.add_argument("preprocessed_dir", type=str)

    parser.add_argument("--denoise", action="store_true", default=False)
    parser.add_argument("--prepare_mfa", action="store_true", default=False)
    parser.add_argument("--mfa", action="store_true", default=False)
    parser.add_argument("--create_dataset", action="store_true", default=False)
    args = parser.parse_args()

    main(args)
